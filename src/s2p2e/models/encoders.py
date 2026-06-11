from __future__ import annotations

import torch
from torch import nn


class PointNetEncoder(nn.Module):
    def __init__(self, input_dim: int = 6, hidden_dim: int = 256, output_dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.mlp(points)
        return features.max(dim=1).values


class TextEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int = 128, output_dim: int = 512) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, output_dim, batch_first=True)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embeds = self.embedding(tokens)
        _, hidden = self.gru(embeds)
        return hidden[-1]


class MLPProjector(nn.Module):
    def __init__(self, input_dim: int, output_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
