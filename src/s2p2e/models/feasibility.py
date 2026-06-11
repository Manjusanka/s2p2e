from __future__ import annotations

import torch
from torch import nn


class FeasibilityFilter(nn.Module):
    def __init__(self, input_dim: int = 8, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, pose: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(pose))
