from __future__ import annotations

import torch
from torch import nn


class TaskWeightRouter(nn.Module):
    def __init__(self, task_weights: dict[str, list[float]], task_names: list[str]) -> None:
        super().__init__()
        matrix = torch.tensor([task_weights[name] for name in task_names], dtype=torch.float32)
        self.register_buffer("weight_table", matrix)

    def forward(self, task_ids: torch.Tensor) -> torch.Tensor:
        return self.weight_table[task_ids]
