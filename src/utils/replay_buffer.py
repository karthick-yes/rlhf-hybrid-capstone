import numpy as np

class ReplayBuffer:
    """
    Stores trajectory segments for preference learning.
    """
    
    def __init__(self, capacity=10000):
        self.capacity = capacity
        self.trajectories = {}  # ID -> (states, actions, cumulative_reward)
        self.next_id = 0
    
    def add_trajectory(self, states, actions, cumulative_reward):
        """
        Add a trajectory segment.
        
        Args:
            states: np.array [T, state_dim]
            actions: np.array [T, action_dim]
            cumulative_reward: float (ground truth for oracle)
        
        Returns:
            traj_id: Assigned ID
        """
        traj_id = self.next_id
        self.trajectories[traj_id] = {
            'states': states,
            'actions': actions,
            'cumulative_reward': cumulative_reward
        }
        self.next_id += 1
        
        # Evict oldest if over capacity
        if len(self.trajectories) > self.capacity:
            oldest_id = min(self.trajectories.keys())
            del self.trajectories[oldest_id]
        
        return traj_id
    
    def get_random_sample(self, n=50):
        """Sample n random trajectory IDs"""
        ids = list(self.trajectories.keys())
        if len(ids) <= n:
            return ids
        return np.random.choice(ids, size=n, replace=False).tolist()
    
    def get_trajectory(self, traj_id):
        """Retrieve trajectory by ID"""
        return self.trajectories[traj_id]
    
    def get_all_ids(self):
        """Get list of all trajectory IDs"""
        return list(self.trajectories.keys())
    
    def __len__(self):
        return len(self.trajectories)