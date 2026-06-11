from __future__ import annotations

import torch
from torch import nn


class PoseToJointIK(nn.Module):
    """Differentiable surrogate IK that maps a pose target to joint references."""

    def __init__(self, pose_dim: int = 8, dof: int = 7, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dof),
        )
        self.qd_head = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dof),
        )
        self.qdd_head = nn.Sequential(
            nn.Linear(pose_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, dof),
        )

    def forward(self, pose: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "q_ref": self.net(pose),
            "qd_ref": self.qd_head(pose) * 0.1,
            "qdd_ref": self.qdd_head(pose) * 0.05,
        }
