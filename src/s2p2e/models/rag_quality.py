from __future__ import annotations

import torch
from torch import nn


class RAGQualityAuditor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 512, output_dim: int = 5, depth: int = 4) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.08),
        ]
        for _ in range(max(depth - 1, 0)):
            layers.extend(
                [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.GELU(),
                    nn.Dropout(0.08),
                ]
            )
        layers.extend([nn.Linear(hidden_dim, output_dim), nn.Sigmoid()])
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)
