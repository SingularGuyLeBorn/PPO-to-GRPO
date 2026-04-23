"""
PPO (Proximal Policy Optimization) 简化实现
Paper: https://arxiv.org/abs/1707.06347

核心特点：
- 需要 Actor + Critic 两个网络
- 使用 GAE 估计优势
- Clipped Surrogate Objective
"""

import torch
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    """策略网络"""
    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        std = self.log_std.exp()
        return mean, std

    def get_log_prob(self, state, action):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

    def sample(self, state):
        mean, std = self.forward(state)
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class Critic(nn.Module):
    """价值网络，估计 V(s)"""
    def __init__(self, state_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        return self.net(state).squeeze()


class PPO:
    def __init__(self, state_dim, action_dim, lr=3e-4, clip_eps=0.2, gamma=0.99, lam=0.95):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.optimizer_a = optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_c = optim.Adam(self.critic.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.gamma = gamma
        self.lam = lam

    def compute_gae(self, rewards, values, dones):
        """计算 Generalized Advantage Estimation"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            next_val = values[t + 1] if t + 1 < len(values) else 0
            delta = rewards[t] + self.gamma * next_val * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        advantages = torch.tensor(advantages)
        returns = advantages + values.detach()
        return advantages, returns

    def update(self, states, actions, old_log_probs, rewards, dones):
        # 1. 用 Critic 估计 V(s)
        values = torch.stack([self.critic(s) for s in states])

        # 2. GAE
        advantages, returns = self.compute_gae(rewards, values, dones)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 3. Actor loss (PPO-Clip)
        log_probs = self.actor.get_log_prob(states, actions)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # 4. Critic loss
        critic_loss = nn.MSELoss()(values, returns)

        # 5. 更新
        self.optimizer_a.zero_grad()
        actor_loss.backward()
        self.optimizer_a.step()

        self.optimizer_c.zero_grad()
        critic_loss.backward()
        self.optimizer_c.step()

        return actor_loss.item(), critic_loss.item()
