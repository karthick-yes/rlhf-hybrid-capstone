"""
Environment wrappers for CartPole and MetaWorld.
Implements Continuous Proxy for Discrete environments to unify SAC architecture.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class SegmentWrapper:
    """
    Wrapper that extracts trajectory segments and handles action conversion.
    """
    
    def __init__(self, env_name, segment_length=50, seed=None):
        self.env_name = env_name
        self.segment_length = segment_length
        
        # 1. Initialize Environment
        if env_name == 'CartPole-v1':
            self.env = gym.make('CartPole-v1', render_mode="rgb_array")
        elif env_name.startswith('metaworld'):
            import metaworld
            task_name = env_name.replace('metaworld-', '')
            if not task_name.endswith('-v2'): task_name += '-v2'
            ml1 = metaworld.ML1(task_name)
            self.env = ml1.train_classes[task_name]()
            task = ml1.train_tasks[0]
            self.env.set_task(task)
        else:
            self.env = gym.make(env_name)
            
        # 2. Unified Action Space (Continuous Proxy)
        # We treat EVERYTHING as continuous to simplify SAC.
        
        if isinstance(self.env.action_space, spaces.Discrete):
            self.is_discrete_wrapped = True
            # Proxy: Agent sees 1 continuous float [-1, 1]
            # We map [-1, 0) -> 0 (Left), [0, 1] -> 1 (Right)
            self.action_dim = 1 
            self.agent_action_dim = 1
            self.action_range = (-1.0, 1.0)
        else:
            self.is_discrete_wrapped = False
            self.action_dim = self.env.action_space.shape[0]
            self.agent_action_dim = self.action_dim
            self.action_range = (
                float(self.env.action_space.low[0]),
                float(self.env.action_space.high[0])
            )

        self.state_dim = self.env.observation_space.shape[0]
        self.is_discrete = False # LIE TO THE AGENT: Always say it's continuous
        
        # 3. Seeding
        if seed is not None:
            self.env.action_space.seed(seed)
            try: self.env.reset(seed=seed)
            except: pass
    
    def reset(self):
        return self.env.reset()
    
    def step(self, action):
        """Step with automatic conversion"""
        # Action comes from SAC as a continuous numpy array (e.g. [-0.85])
        
        if self.is_discrete_wrapped:
            # Continuous Proxy -> Discrete
            # -1.0 to 0.0 -> Action 0
            #  0.0 to 1.0 -> Action 1
            scalar_action = action.item() if isinstance(action, np.ndarray) else action
            env_action = 1 if scalar_action > 0 else 0
        else:
            # Standard Continuous
            env_action = action
            
        return self.env.step(env_action)
    
    def collect_segment(self, policy, max_steps=None):
        """Collect data using the policy"""
        if max_steps is None: max_steps = self.segment_length
        
        states, actions, rewards = [], [], []
        state, _ = self.reset()
        done = False
        steps = 0
        
        while not done and steps < max_steps:
            # Policy outputs continuous action
            action = policy(state)
            
            # Step converts it internally
            next_state, reward, term, trunc, _ = self.step(action)
            done = term or trunc
            
            states.append(state)
            actions.append(action) # Store the CONTINUOUS action for training
            rewards.append(reward)
            
            state = next_state
            steps += 1
        
        return np.array(states), np.array(actions), np.sum(rewards)

def make_env(env_name, segment_length=50, seed=None):
    return SegmentWrapper(env_name, segment_length, seed)