import numpy as np

class Oracle:
    """
    Simulated human teacher that provides preferences based on ground truth rewards.
    """
    def __init__(self, env_name='CartPole-v1'):
        self.env_name = env_name
    
    def compare(self, segment_1, segment_2):
    # Handle dict from replay buffer
        if isinstance(segment_1, dict):
            r1 = segment_1['cumulative_reward']
        elif isinstance(segment_1, (list, np.ndarray)):
            r1 = np.sum(segment_1)
        else:
            r1 = segment_1
    
        if isinstance(segment_2, dict):
            r2 = segment_2['cumulative_reward']
        elif isinstance(segment_2, (list, np.ndarray)):
            r2 = np.sum(segment_2)
        else:
            r2 = segment_2
    
        return 1 if r1 > r2 else 0