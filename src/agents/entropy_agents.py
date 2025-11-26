import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import NearestNeighbors


class EntropyAgent:
    """
    Agent that maximizes state entropy for diverse exploration.
    
    Implements PEBBLE-style unsupervised pre-training:
    - k-NN entropy estimation for intrinsic rewards
    - Policy gradient to maximize expected entropy
    - Compatible with SAC's maximum entropy framework
    """
    
    def __init__(self, state_dim, action_dim, action_range, k=5, device='cpu'):
        """
        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            action_range: (low, high) tuple for action clipping
            k: Number of nearest neighbors for entropy estimation
            device: 'cpu' or 'cuda'
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_range = action_range
        self.k = k
        self.device = device
        
        # Stochastic policy network (Gaussian)
        # Output: mean and log_std for action distribution
        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        ).to(device)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(256, action_dim).to(device)
        self.log_std_head = nn.Linear(256, action_dim).to(device)
        
        # Initialize log_std to small values for initial exploration
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.log_std_head.bias, -1e-3, 1e-3)
        
        self.optimizer = optim.Adam(
            list(self.policy_net.parameters()) + 
            list(self.mean_head.parameters()) + 
            list(self.log_std_head.parameters()),
            lr=3e-4
        )
        
        # State buffer for k-NN entropy estimation
        self.state_buffer = []
        self.max_buffer_size = 10000
        
        # Constants for numerical stability
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2
    
    def select_action(self, state, deterministic=False):
        """
        Select action from stochastic policy.
        
        Args:
            state: np.array [state_dim]
            deterministic: If True, return mean action (for evaluation)
        
        Returns:
            action: np.array [action_dim]
            log_prob: float (for policy gradient, optional)
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.policy_net(state_t)
            mean = self.mean_head(features)
            log_std = self.log_std_head(features)
            log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
            std = log_std.exp()
        
        if deterministic:
            # For evaluation: use mean action
            action = torch.tanh(mean).cpu().numpy()[0]
        else:
            # Sample from Gaussian distribution
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t).cpu().numpy()[0]
        
        # Scale to action range
        action_scaled = action * (self.action_range[1] - self.action_range[0]) / 2.0
        action_scaled = action_scaled + (self.action_range[1] + self.action_range[0]) / 2.0
        
        return action_scaled
    
    def add_state(self, state):
        """Add state to buffer for k-NN computation"""
        self.state_buffer.append(state.copy())
        if len(self.state_buffer) > self.max_buffer_size:
            # Remove oldest states (FIFO)
            self.state_buffer.pop(0)
    
    def compute_intrinsic_reward(self, state):
        """
        Compute k-NN entropy as intrinsic reward.
        
        Formula: r_int(s) = log ||s - s^(k)||_2
        
        Higher reward for states far from existing data (encourages exploration).
        
        Args:
            state: np.array [state_dim]
        
        Returns:
            reward: float
        """
        if len(self.state_buffer) < self.k + 1:
            # Not enough data: encourage exploration uniformly
            return 1.0
        
        try:
            # Build k-NN index
            nbrs = NearestNeighbors(n_neighbors=min(self.k+1, len(self.state_buffer)), 
                                   algorithm='ball_tree')
            nbrs.fit(self.state_buffer)
            
            # Find distances to k nearest neighbors
            distances, _ = nbrs.kneighbors([state])
            
            # distances[0][0] is self (distance=0), use k-th neighbor
            if len(distances[0]) > self.k:
                kth_distance = distances[0][self.k]
            else:
                kth_distance = distances[0][-1]
            
            # Intrinsic reward = log(distance + epsilon)
            reward = np.log(kth_distance + 1e-6)
            
            # Clip to reasonable range to prevent instability
            reward = np.clip(reward, -10.0, 10.0)
            
            return reward
            
        except Exception as e:
            print(f"Warning: k-NN failed with {e}, returning default reward")
            return 1.0
    
    def update_policy(self, states, intrinsic_rewards):
        """
        Update policy to maximize expected intrinsic rewards + entropy.
        
        Uses REINFORCE-style policy gradient with entropy bonus:
        L = -E[r_int + α * H(π)]
        
        Where H(π) is the entropy of the action distribution.
        
        Args:
            states: np.array [batch_size, state_dim]
            intrinsic_rewards: np.array [batch_size]
        
        Returns:
            loss: float
            entropy: float (for logging)
        """
        if len(states) < 2:
            return 0.0, 0.0
        
        states_t = torch.FloatTensor(states).to(self.device)
        rewards_t = torch.FloatTensor(intrinsic_rewards).to(self.device)
        
        # Normalize rewards for stability (optional but recommended)
        rewards_t = (rewards_t - rewards_t.mean()) / (rewards_t.std() + 1e-8)
        
        # Forward pass through policy
        features = self.policy_net(states_t)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        
        # Create action distribution
        normal = torch.distributions.Normal(mean, std)
        
        # Sample actions (reparameterization for gradients)
        x_t = normal.rsample()
        actions = torch.tanh(x_t)
        
        # Compute log probability with tanh squashing correction
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(u))
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - actions.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        
        # Compute entropy (higher is better for exploration)
        entropy = normal.entropy().sum(dim=1).mean()
        
        # Policy gradient loss
        # Maximize: E[r_int] + α * H(π)
        # Minimize: -E[r_int] - α * H(π)
        alpha = 0.1  # Entropy coefficient (can be tuned)
        policy_loss = -(rewards_t.unsqueeze(1) * log_prob).mean()
        entropy_loss = -alpha * entropy
        
        total_loss = policy_loss + entropy_loss
        
        # Backward pass with gradient clipping
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + 
            list(self.mean_head.parameters()) + 
            list(self.log_std_head.parameters()),
            max_norm=1.0  # Prevents gradient explosion
        )
        self.optimizer.step()
        
        return total_loss.item(), entropy.item()


class SimpleEntropySAC:
    """
    Simplified wrapper for entropy-driven exploration.
    Compatible with Phase 2/3 SAC implementation.
    """
    
    def __init__(self, state_dim, action_dim, action_range, k=5, device='cpu'):
        self.agent = EntropyAgent(state_dim, action_dim, action_range, k, device)
        self.device = device
    
    def collect_trajectory(self, env, max_steps=200):
        """
        Collect one trajectory using entropy-driven exploration.
        
        Args:
            env: Gym environment
            max_steps: Maximum trajectory length
        
        Returns:
            states: np.array [T, state_dim]
            actions: np.array [T, action_dim]
            intrinsic_rewards: np.array [T] - For entropy maximization
            extrinsic_rewards: np.array [T] - For buffer/quality check
            episode_length: int
        """
        states_list = []
        actions_list = []
        intrinsic_rewards_list = []
        extrinsic_rewards_list = []
        
        # Reset environment
        result = env.reset()
        state = result[0] if isinstance(result, tuple) else result
        
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Select action from policy
            action = self.agent.select_action(state, deterministic=False)
            
            # Step environment
            if hasattr(env, 'action_space') and hasattr(env.action_space, 'n'):
                # Discrete environment: convert continuous to discrete
                action_discrete = int(action[0] > 0.5) if isinstance(action, np.ndarray) else int(action > 0.5)
                step_result = env.step(action_discrete)
            else:
                # Continuous environment: use as-is
                step_result = env.step(action)
                
            next_state = step_result[0]
            extrinsic_reward = step_result[1] # REAL REWARD
            done = step_result[2] if len(step_result) > 2 else False
            
            # Compute intrinsic reward BEFORE updating buffer
            intrinsic_reward = self.agent.compute_intrinsic_reward(state)
            
            # Store transition
            states_list.append(state)
            actions_list.append(action)
            intrinsic_rewards_list.append(intrinsic_reward)
            extrinsic_rewards_list.append(extrinsic_reward)
            
            # Update buffer with current state
            self.agent.add_state(state)
            
            state = next_state
            steps += 1
        
        # Convert to numpy arrays
        states = np.array(states_list)
        actions = np.array(actions_list)
        int_rewards = np.array(intrinsic_rewards_list)
        ext_rewards = np.array(extrinsic_rewards_list)
        
        return states, actions, int_rewards, ext_rewards, len(states_list)
    
    def warmup(self, env, n_episodes=50, update_freq=1):
        """
        Run warmup exploration phase.
        
        Args:
            env: Gym environment
            n_episodes: Number of episodes to collect
            update_freq: Update policy every N episodes
        
        Returns:
            trajectories: List of (states, actions, int_rewards, ext_rewards) tuples
            metrics: Dict with training statistics
        """
        print(f"\n Starting PEBBLE-style Entropy Warmup ({n_episodes} episodes)...")
        
        trajectories = []
        metrics = {
            'episode_lengths': [],
            'avg_intrinsic_rewards': [],
            'policy_losses': [],
            'entropies': []
        }
        
        for ep in range(n_episodes):
            # Collect trajectory with both rewards
            states, actions, int_rewards, ext_rewards, ep_len = self.collect_trajectory(env)
            trajectories.append((states, actions, int_rewards, ext_rewards))
            
            # Track metrics (using intrinsic rewards)
            metrics['episode_lengths'].append(ep_len)
            metrics['avg_intrinsic_rewards'].append(int_rewards.mean())
            
            # Update policy periodically (using intrinsic rewards)
            if (ep + 1) % update_freq == 0 and len(states) > 10:
                loss, entropy = self.agent.update_policy(states, int_rewards)
                metrics['policy_losses'].append(loss)
                metrics['entropies'].append(entropy)
            
            # Logging
            if (ep + 1) % 10 == 0:
                recent_lengths = metrics['episode_lengths'][-10:]
                recent_rewards = metrics['avg_intrinsic_rewards'][-10:]
                recent_entropy = metrics['entropies'][-5:] if metrics['entropies'] else [0]
                
                print(f"   Episode {ep+1}/{n_episodes} | "
                      f"Avg Length: {np.mean(recent_lengths):.1f} | "
                      f"Avg Reward: {np.mean(recent_rewards):.3f} | "
                      f"Entropy: {np.mean(recent_entropy):.3f}")
        
        print(f" Warmup Complete: {len(trajectories)} diverse trajectories collected")
        print(f"   Total states in buffer: {len(self.agent.state_buffer)}")
        print(f"   Average episode length: {np.mean(metrics['episode_lengths']):.1f}\n")
        
        return trajectories, metrics


# ============================================================================
# PHASE 1 VALIDATION SCRIPT
# ===========================================================================

def validate_phase1():
    """
    Checkpoint: Verify entropy warmup works correctly.
    """
    print("="*70)
    print("PHASE 1 VALIDATION: Entropy Warmup")
    print("="*70)
    
    try:
        import gymnasium as gym
        
        # Test on CartPole-v1
        env = gym.make('CartPole-v1')
        state_dim = env.observation_space.shape[0]
        action_dim = 1  # Discrete but treated as continuous
        action_range = (0, 1)
        
        # Initialize entropy agent
        agent = SimpleEntropySAC(
            state_dim=state_dim,
            action_dim=action_dim,
            action_range=action_range,
            k=5,
            device='gpu'
        )
        
        # Run warmup
        trajectories, metrics = agent.warmup(env, n_episodes=20, update_freq=5)
        
        # Validation checks
        print("\n" + "="*70)
        print("VALIDATION RESULTS:")
        print("="*70)
        
        avg_length = np.mean(metrics['episode_lengths'])
        print(f"✓ Average episode length: {avg_length:.1f} steps")
        
        if avg_length > 50:
            print("  → PASS: Episodes are long enough (> 50 steps)")
        else:
            print("  → WARNING: Episodes are short (< 50 steps)")
        
        buffer_size = len(agent.agent.state_buffer)
        print(f"✓ Buffer size: {buffer_size} states")
        
        if buffer_size > 100:
            print("  → PASS: Sufficient state coverage")
        else:
            print("  → WARNING: Low state coverage")
        
        if metrics['entropies']:
            avg_entropy = np.mean(metrics['entropies'])
            print(f"✓ Average policy entropy: {avg_entropy:.3f}")
            
            if avg_entropy > 0:
                print("  → PASS: Policy maintains exploration")
            else:
                print("  → WARNING: Low entropy (may converge prematurely)")
        
        # State diversity check
        if buffer_size > 10:
            states_array = np.array(agent.agent.state_buffer)
            state_std = states_array.std(axis=0).mean()
            print(f"✓ State diversity (avg std): {state_std:.3f}")
            
            if state_std > 0.1:
                print("  → PASS: States are diverse")
            else:
                print("  → WARNING: Low state diversity")
        
        print("\n" + "="*70)
        print("PHASE 1 VALIDATION COMPLETE")
        print("="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ VALIDATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    validate_phase1()