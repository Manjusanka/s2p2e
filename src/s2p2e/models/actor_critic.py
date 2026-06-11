from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, depth: int = 3) -> None:
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(depth - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsInjectedActor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, physics_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_encoder = MLP(obs_dim, hidden_dim, hidden_dim)
        self.actor = MLP(hidden_dim + physics_dim, hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor, physics_feat: torch.Tensor) -> torch.Tensor:
        obs_feat = self.obs_encoder(obs)
        return torch.tanh(self.actor(torch.cat([obs_feat, physics_feat], dim=-1)))


class PhysicsInjectedCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, physics_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.obs_encoder = MLP(obs_dim + action_dim, hidden_dim, hidden_dim)
        self.critic = MLP(hidden_dim + physics_dim, hidden_dim, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor, physics_feat: torch.Tensor) -> torch.Tensor:
        obs_feat = self.obs_encoder(torch.cat([obs, action], dim=-1))
        return self.critic(torch.cat([obs_feat, physics_feat], dim=-1))


class PhysicsInjectedActorCritic(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, physics_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.actor = PhysicsInjectedActor(obs_dim, action_dim, physics_dim, hidden_dim)
        self.critic = PhysicsInjectedCritic(obs_dim, action_dim, physics_dim, hidden_dim)
        self.target_critic = PhysicsInjectedCritic(obs_dim, action_dim, physics_dim, hidden_dim)
        self.target_critic.load_state_dict(self.critic.state_dict())

    def act(self, obs: torch.Tensor, physics_feat: torch.Tensor) -> torch.Tensor:
        return self.actor(obs, physics_feat)

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor, physics_feat: torch.Tensor) -> torch.Tensor:
        return self.critic(obs, action, physics_feat)

    @staticmethod
    def shape_reward(
        base_reward: torch.Tensor,
        predicted_tau: torch.Tensor,
        torque_limit: float,
        lambda_viol: float,
    ) -> torch.Tensor:
        penalty = torch.relu(predicted_tau.abs() - torque_limit).mean(dim=-1, keepdim=True)
        return base_reward - lambda_viol * penalty

    @staticmethod
    def contrastive_loss(feasible_feat: torch.Tensor, infeasible_feat: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        positive = F.cosine_similarity(feasible_feat, feasible_feat, dim=-1).exp() / temperature
        negative = F.cosine_similarity(feasible_feat, infeasible_feat, dim=-1).exp() / temperature
        return (-torch.log(positive / (positive + negative + 1e-6))).mean()

    def soft_update(self, tau: float = 0.005) -> None:
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
