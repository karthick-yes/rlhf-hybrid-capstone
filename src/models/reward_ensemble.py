
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class RewardNetwork(nn.Module):
    """Single reward model in the ensemble"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(self, state, action):
        """
        Args:
            state: Tensor [Batch, ...State_dims]
            action: Tensor [Batch, ...Action_dims]
        Returns:
            reward: Tensor [Batch, 1]
        """
        # Ensure state and action have compatible shapes
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
            
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class RewardEnsemble:
    """
    Deep Ensemble for reward prediction with uncertainty quantification.
    
    Features:
    - Automatic GPU detection (uses CUDA if available)
    - Handles both trajectory segments [Batch, Time, Dim] and single steps [Batch, Dim]
    - Gradient clipping for stability
    - Independent optimizers with weight decay for diversity
    """
    
    def __init__(self, state_dim, action_dim, config=None, device=None):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            config: Dict with 'ensemble_size', 'lr', etc. (optional)
            device: 'cpu', 'cuda', or None (auto-detect)
        """
        # Default config
        if config is None:
            config = {}
        
        self.K = config.get('ensemble_size', 5)
        self.lr = float(config.get('lr', 3e-4))
        self.weight_decay = float(config.get('weight_decay', 1e-4))
        self.hidden_dim = config.get('hidden_dim', 256)
        
        # Device handling: auto-detect if not specified
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"[RewardEnsemble] Using device: {self.device}")
        
        # Create ensemble
        self.models = []
        self.optimizers = []
        
        for i in range(self.K):
            model = RewardNetwork(state_dim, action_dim, self.hidden_dim)
            model = model.to(self.device)  # Move to device
            
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.lr,
                weight_decay=self.weight_decay
            )
            
            self.models.append(model)
            self.optimizers.append(optimizer)
        
        print(f"[RewardEnsemble] Initialized {self.K} models on {self.device}")
    
    def _to_device(self, *tensors):
        """Helper to move tensors to device"""
        return tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t 
                    for t in tensors)
    
    def _compute_segment_reward(self, model, states, actions):
        """
        Compute cumulative reward for a trajectory segment.
        
        Handles both:
        - Segments: [Batch, Time, Dim] -> sum over Time
        - Single steps: [Batch, Dim] -> no summation
        
        Args:
            model: RewardNetwork instance
            states: Tensor [Batch, Time?, State_dim]
            actions: Tensor [Batch, Time?, Action_dim]
        
        Returns:
            reward: Tensor [Batch, 1]
        """
        if states.dim() == 3:  # Trajectory segment [Batch, Time, Dim]
            # Process each timestep and sum
            batch_size, seq_len, _ = states.shape
            rewards = []
            
            for t in range(seq_len):
                r_t = model(states[:, t, :], actions[:, t, :])
                rewards.append(r_t)
            
            # Sum over time: [Batch, Time, 1] -> [Batch, 1]
            cumulative = torch.stack(rewards, dim=1).sum(dim=1)
            return cumulative
        
        else:  # Single step [Batch, Dim]
            return model(states, actions)
    
    def train_step(self, s0, a0, s1, a1, labels):
        """
        Train ensemble on a batch of preference pairs.
        
        Convention:
            label=1 means s1 preferred over s0 (s1 > s0)
            label=0 means s0 preferred over s1 (s0 > s1)
        
        Args:
            s0, a0: Segment 0 states/actions [Batch, Time?, State/Action_dim]
            s1, a1: Segment 1 states/actions [Batch, Time?, State/Action_dim]
            labels: Tensor [Batch] with values 0 or 1
        
        Returns:
            avg_loss: Average loss across ensemble
        """
        # Move to device
        s0, a0, s1, a1, labels = self._to_device(s0, a0, s1, a1, labels)
        
        # Ensure labels are float and shaped [Batch, 1]
        labels = labels.float().unsqueeze(1) if labels.dim() == 1 else labels.float()
        
        #smoothing_factor = self.config.get('label_smoothing', 0.05)
        #labels = labels * (1 - 2 * smoothing_factor) + smoothing_factor
        
        total_loss = 0.0
        criterion = nn.BCEWithLogitsLoss()
        
        for i in range(self.K):
            self.optimizers[i].zero_grad()
            
            # Compute cumulative rewards for both segments
            r0 = self._compute_segment_reward(self.models[i], s0, a0)
            r1 = self._compute_segment_reward(self.models[i], s1, a1)
            
            # Bradley-Terry-Luce model:
            # P(s1 > s0) = sigmoid(r1 - r0)
            # If label=1, we want r1 > r0 (positive logit)
            logits = r1 - r0
            
            loss = criterion(logits, labels)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(
                self.models[i].parameters(),
                max_norm=10.0
            )
            
            self.optimizers[i].step()
            total_loss += loss.item()
        
        return total_loss / self.K
    
    def predict_segment(self, states, actions):
        """
        Predict mean and std for trajectory segment(s).
        
        Args:
            states: Tensor [Batch, Time, State_dim] or [Batch, State_dim]
            actions: Tensor [Batch, Time, Action_dim] or [Batch, Action_dim]
        
        Returns:
            mean: Tensor [Batch, 1] - ensemble mean
            std: Tensor [Batch, 1] - ensemble std (epistemic uncertainty)
        """
        states, actions = self._to_device(states, actions)
        
        with torch.no_grad():
            predictions = []
            
            for model in self.models:
                r = self._compute_segment_reward(model, states, actions)
                predictions.append(r)
            
            # Stack: [K, Batch, 1]
            predictions = torch.stack(predictions, dim=0)
            
            # Compute statistics across ensemble
            mean = predictions.mean(dim=0)  # [Batch, 1]
            std = predictions.std(dim=0)    # [Batch, 1]
            
            return mean, std
    
    def predict_step(self, state, action):
        """
        Predict reward for single state-action pair (for RL agent).
        
        Args:
            state: Tensor [State_dim] or [Batch, State_dim]
            action: Tensor [Action_dim] or [Batch, Action_dim]
        
        Returns:
            mean: Tensor [1] or [Batch, 1]
            std: Tensor [1] or [Batch, 1]
        """
        state, action = self._to_device(state, action)
        
        # Ensure batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False
        
        with torch.no_grad():
            predictions = []
            
            for model in self.models:
                r = model(state, action)
                predictions.append(r)
            
            predictions = torch.stack(predictions, dim=0)  # [K, Batch, 1]
            mean = predictions.mean(dim=0)
            std = predictions.std(dim=0)
            
            if squeeze_output:
                mean = mean.squeeze(0)
                std = std.squeeze(0)
            
            return mean, std
    
    def save(self, path):
        """Save ensemble to disk"""
        checkpoint = {
            'models': [model.state_dict() for model in self.models],
            'optimizers': [opt.state_dict() for opt in self.optimizers],
            'config': {
                'K': self.K,
                'lr': self.lr,
                'weight_decay': self.weight_decay,
                'hidden_dim': self.hidden_dim
            }
        }
        torch.save(checkpoint, path)
        print(f"[RewardEnsemble] Saved to {path}")
    
    def load(self, path):
        """Load ensemble from disk"""
        checkpoint = torch.load(path, map_location=self.device)
        
        for i in range(self.K):
            self.models[i].load_state_dict(checkpoint['models'][i])
            self.optimizers[i].load_state_dict(checkpoint['optimizers'][i])
        
        print(f"[RewardEnsemble] Loaded from {path}")
    
    def predict_uncertainty(self, state, action):
        """
        Legacy method for backward compatibility with test files.
        Returns only std (uncertainty), not mean.
        
        Args:
            state: Tensor [Batch, Time?, Dim]
            action: Tensor [Batch, Time?, Dim]
        
        Returns:
            std: Tensor [Batch, 1] - epistemic uncertainty
        """
        _, std = self.predict_segment(state, action)
        return std
    
    def to(self, device):
        """Move all models to specified device"""
        self.device = torch.device(device)
        for i in range(self.K):
            self.models[i] = self.models[i].to(self.device)
        print(f"[RewardEnsemble] Moved to {self.device}")