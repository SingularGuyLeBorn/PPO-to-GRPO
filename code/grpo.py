"""
GRPO (Group Relative Policy Optimization) 简化实现
Paper: DeepSeekMath (2024) - https://arxiv.org/abs/2402.03300

核心特点：
- 去掉 Critic 网络，只用 Actor
- 组内采样 -> Z-Score 归一化作为优势
- 适合 LLM 对齐场景
"""

import torch
import torch.nn as nn
import torch.optim as optim


class GRPOActor(nn.Module):
    """策略网络（模拟 LLM 生成）"""
    def __init__(self, vocab_size=100, embed_dim=64, hidden_dim=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids):
        x = self.embed(input_ids)
        out, _ = self.rnn(x)
        logits = self.head(out)
        return logits

    def get_log_prob(self, input_ids, response_ids):
        """计算给定响应的 log probability"""
        logits = self.forward(input_ids)
        # 取最后一个 token 的 logits 做采样（简化）
        last_logits = logits[:, -1, :]
        dist = torch.distributions.Categorical(logits=last_logits)
        return dist.log_prob(response_ids)

    def sample(self, prompt_ids, max_len=10):
        """从 prompt 采样一个 response"""
        self.eval()
        with torch.no_grad():
            logits = self.forward(prompt_ids)
            last_logits = logits[:, -1, :]
            dist = torch.distributions.Categorical(logits=last_logits)
            response = dist.sample()
            log_prob = dist.log_prob(response)
        return response, log_prob


class GRPO:
    """
    GRPO 训练器

    核心思想：
    1. 对同一个 prompt 采样 N 个回答（组）
    2. 用 Reward Model 给每个回答打分
    3. 组内 Z-Score 归一化作为优势估计
    4. 用 PPO-Clip 类似的 clipped 目标更新策略
    5. 加 KL 散度正则防止偏离参考策略太远
    """
    def __init__(self, actor, ref_actor, lr=3e-4, clip_eps=0.2, group_size=8, kl_coef=0.01):
        self.actor = actor
        self.ref_actor = ref_actor  # 参考策略（冻结）
        self.optimizer = optim.Adam(actor.parameters(), lr=lr)
        self.clip_eps = clip_eps
        self.group_size = group_size
        self.kl_coef = kl_coef

        # 冻结参考策略
        for p in self.ref_actor.parameters():
            p.requires_grad = False

    def compute_kl(self, prompt, responses):
        """计算当前策略 vs 参考策略的 KL 散度"""
        kl_sum = 0
        for r in responses:
            log_p = self.actor.get_log_prob(prompt, r)
            log_ref = self.ref_actor.get_log_prob(prompt, r)
            # 近似 KL(p||q) = sum(p * (log_p - log_q))
            p = log_p.exp().detach()
            kl_sum += (p * (log_p - log_ref)).sum()
        return kl_sum / len(responses)

    def update(self, prompt, reward_fn):
        """
        单步 GRPO 更新

        Args:
            prompt: 输入的 prompt token ids
            reward_fn: 奖励函数/模型，输入 response 返回标量奖励
        """
        # 1. 组内采样 group_size 个回答
        responses = []
        log_probs_list = []
        for _ in range(self.group_size):
            response, log_prob = self.actor.sample(prompt)
            responses.append(response)
            log_probs_list.append(log_prob)

        # 2. 计算奖励（模拟 Reward Model）
        rewards = torch.tensor([
            reward_fn(r) for r in responses
        ], dtype=torch.float32)

        # 3. 组内 Z-Score 归一化 -> 优势
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # 4. GRPO Clipped Objective
        log_probs = torch.stack(log_probs_list)
        old_log_probs = log_probs.detach()
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # 5. KL 散度正则
        kl_loss = self.compute_kl(prompt, responses)

        # 6. 总损失
        total_loss = policy_loss + self.kl_coef * kl_loss

        # 7. 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "policy_loss": policy_loss.item(),
            "kl_loss": kl_loss.item(),
            "mean_reward": rewards.mean().item(),
        }
