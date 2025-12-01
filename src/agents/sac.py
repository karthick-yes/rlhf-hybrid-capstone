import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

# Standard Continuous SAC Components

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        torch.nn.init.uniform_(self.mean.weight, -1e-3, 1e-3)
        torch.nn.init.uniform_(self.mean.bias, -1e-3, 1e-3)
        
        
        
    def forward(self, x):
        x = self.base(x)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2) # Stability clamping
        return mean, log_std
        
    def sample(self, x):
        mean, log_std = self.forward(x)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample() # Reparameterization
        action = torch.tanh(x_t)
        
        # Enforce action bounds logic in log_prob
        log_prob = dist.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(1, keepdim=True)

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        return self.q1(x), self.q2(x)

class SAC:
    """
    Standard Continuous SAC.
    Relies on the Environment Wrapper to handle discrete conversions.
    """
    def __init__(self, state_dim, action_dim, is_discrete=False, device='cpu', lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.lr = lr  
        
        # Ignore is_discrete flag - we treat everything as continuous now
        
        self.actor = Actor(state_dim, action_dim).to(device)
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state, deterministic=False):
        state_t = torch.FloatTensor(state).to(self.device)
        if len(state.shape) == 1: state_t = state_t.unsqueeze(0)
            
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_t)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state_t)
                
        return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=64):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 1. Update Critic
        with torch.no_grad():
            next_actions, next_log_probs = self.actor.sample(next_states)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = torch.min(q1_target, q2_target) - self.alpha * next_log_probs
            target = rewards + (1 - dones) * self.gamma * q_target
        
        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # 2. Update Actor
        # Freeze critic
        for p in self.critic.parameters(): p.requires_grad = False
        
        new_actions, log_probs = self.actor.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()
        
        # Unfreeze critic
        for p in self.critic.parameters(): p.requires_grad = True

        # 3. Soft Update
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return actor_loss.item(), critic_loss.item()
    
    def reset_critic(self):
        print("   [SAC] Resetting Critic Weights...")
        # Re-initialize Critic networks
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Re-initialize Optimizer
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)