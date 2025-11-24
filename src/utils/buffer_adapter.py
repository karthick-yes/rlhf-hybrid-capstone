import numpy as np
import torch

class SACBufferAdapter:
    """
    Bridging the gap between Preference Learning (Trajectories) 
    and Reinforcement Learning (Transitions).
    
    Performs Phase 3: Reward Relabeling.
    """
    def __init__(self, state_dim, action_dim, capacity=100000, device='cpu'):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.capacity = capacity
        self.device = device
        
        # Pre-allocate memory for SAC transitions
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32) # These will be LEARNED rewards
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        
        self.ptr = 0
        self.size = 0

    def relabel_and_flatten(self, trajectory_buffer, reward_ensemble):
        """
        1. Extract all trajectories from the Preference Buffer.
        2. Use the Ensemble to predict rewards for every step.
        3. Store as transitions (s, a, r_learned, s', d) for SAC.
        """
        print("   [Adapter] Relabeling buffer with learned rewards...")
        self.ptr = 0
        self.size = 0
        
        all_ids = trajectory_buffer.get_all_ids()
        
        for tid in all_ids:
            traj = trajectory_buffer.get_trajectory(tid)
            states = traj['states']     # Shape: [T, dim]
            actions = traj['actions']   # Shape: [T, dim]
            
            # Convert to tensor for reward prediction
            s_t = torch.FloatTensor(states).to(self.device)
            a_t = torch.FloatTensor(actions).to(self.device)
            
            # --- PREDICT REWARD ---
            # We assume the ensemble has a predict_reward method that returns (mean, std)
            # If not, we manually query the models
            with torch.no_grad():
                preds = torch.stack([m(s_t, a_t) for m in reward_ensemble.models])
                # Use Mean - k * Std (LCB) for robust RL, or just Mean
                # Using Mean for standard RLHF
                learned_rewards = preds.mean(dim=0).cpu().numpy() # Shape: [T, 1]
            
            # --- FLATTEN LOOP ---
            T = len(states)
            for t in range(T - 1):
                self.add(
                    states[t], 
                    actions[t], 
                    learned_rewards[t], # The NEW learned reward
                    states[t+1], 
                    0.0 if t < T-2 else 1.0 # Simple done logic
                )
                
        print(f"   [Adapter] Flattened into {self.size} transitions for SAC.")

    def add(self, state, action, reward, next_state, done):
        """Standard cyclic buffer add"""
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """Standard random sampling for SAC"""
        ind = np.random.randint(0, self.size, size=batch_size)
        return (
            self.states[ind],
            self.actions[ind],
            self.rewards[ind],
            self.next_states[ind],
            self.dones[ind]
        )