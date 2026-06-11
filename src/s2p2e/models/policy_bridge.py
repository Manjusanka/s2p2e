from __future__ import annotations

import torch
from torch import nn

from s2p2e.models.actor_critic import PhysicsInjectedActorCritic


class PPEPolicyBridge(nn.Module):
    """Bridge PPE latent features into an actor-critic policy."""

    def __init__(self, obs_dim: int, action_dim: int, ppe_feature_dim: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.physics_proj = nn.Sequential(
            nn.Linear(ppe_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.policy = PhysicsInjectedActorCritic(obs_dim=obs_dim, action_dim=action_dim, physics_dim=hidden_dim, hidden_dim=hidden_dim)

    def act(self, obs: torch.Tensor, ppe_features: torch.Tensor) -> torch.Tensor:
        return self.policy.act(obs, self.physics_proj(ppe_features))

    def evaluate(self, obs: torch.Tensor, action: torch.Tensor, ppe_features: torch.Tensor) -> torch.Tensor:
        return self.policy.evaluate(obs, action, self.physics_proj(ppe_features))
