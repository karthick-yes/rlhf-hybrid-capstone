"""
Environment wrappers for CartPole and MetaWorld.
"""

import numpy as np
import gymnasium as gym

class SegmentWrapper:
    """
    Wrapper that extracts trajectory segments for preference learning.
    """
    
    def __init__(self, env_name, segment_length=50, seed=None):
        """
        Args:
            env_name: 'CartPole-v1' or 'metaworld-door-open'
            segment_length: Length of trajectory segments
            seed: Random seed
        """
        self.env_name = env_name
        self.segment_length = segment_length
        
        if env_name == 'CartPole-v1':
            self.env = gym.make('CartPole-v1')
            self.state_dim = 4
            self.action_dim = 1  # Discrete, but we'll treat as continuous
            self.action_range = (0, 1)
        elif env_name.startswith('metaworld'):
            # MetaWorld setup
            import metaworld
            ml1 = metaworld.ML1('door-open-v2')
            self.env = ml1.train_classes['door-open-v2']()
            task = ml1.train_tasks[0]
            self.env.set_task(task)
            self.state_dim = self.env.observation_space.shape[0]
            self.action_dim = self.env.action_space.shape[0]
            self.action_range = (
                self.env.action_space.low[0],
                self.env.action_space.high[0]
            )
        else:
            raise ValueError(f"Unknown environment: {env_name}")
        
        if seed is not None:
            self.env.action_space.seed(seed)
            if hasattr(self.env, 'reset'):
                self.env.reset(seed=seed)
    
    def reset(self):
        """Reset environment"""
        result = self.env.reset()
        if isinstance(result, tuple):
            return result[0], result[1]
        return result, {}
    
    def step(self, action):
        """Step environment"""
        # Handle discrete CartPole
        if self.env_name == 'CartPole-v1':
            action = int(action > 0.5)
        
        return self.env.step(action)
    
    def collect_segment(self, policy, max_steps=None):
        """
        Collect a trajectory segment using given policy.
        
        Args:
            policy: Function state -> action
            max_steps: Maximum steps (defaults to segment_length)
        
        Returns:
            states: np.array [T, state_dim]
            actions: np.array [T, action_dim]
            cumulative_reward: float
        """
        if max_steps is None:
            max_steps = self.segment_length
        
        states_list = []
        actions_list = []
        rewards_list = []
        
        state, _ = self.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Get action from policy
            action = policy(state)
            
            # Step
            next_state, reward, terminated, truncated, info = self.step(action)
            done = terminated or truncated
            
            # Store
            states_list.append(state)
            actions_list.append([action] if np.isscalar(action) else action)
            rewards_list.append(reward)
            
            state = next_state
            steps += 1
        
        states = np.array(states_list)
        actions = np.array(actions_list)
        cumulative_reward = np.sum(rewards_list)
        
        return states, actions, cumulative_reward
    
    def collect_random_segment(self):
        """Collect segment with random actions"""
        return self.collect_segment(
            policy=lambda s: self.env.action_space.sample()
        )


def make_env(env_name, segment_length=50, seed=None):
    """Factory function to create wrapped environment"""
    return SegmentWrapper(env_name, segment_length, seed)