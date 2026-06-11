from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FiLMLayer(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int) -> None:
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta = nn.Linear(cond_dim, hidden_dim)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma = self.gamma(cond)
        beta = self.beta(cond)
        return x * (1.0 + gamma) + beta


class PhysicsPriorEncoder(nn.Module):
    def __init__(self, state_dim: int, cond_dim: int, dof: int, hidden_dim: int = 512, layers: int = 4) -> None:
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dim)
        self.hidden = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers - 1))
        self.films = nn.ModuleList(FiLMLayer(hidden_dim, cond_dim) for _ in range(layers))
        self.output = nn.Linear(hidden_dim, dof)
        self.feasibility = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, embodiment: torch.Tensor) -> dict[str, torch.Tensor]:
        x = F.relu(self.films[0](self.input_layer(state), embodiment))
        for layer, film in zip(self.hidden, self.films[1:]):
            x = F.relu(film(layer(x), embodiment))
        tau = self.output(x)
        feasible = torch.sigmoid(self.feasibility(x))
        return {"tau": tau, "features": x, "feasible": feasible}

    @staticmethod
    def loss(
        outputs: dict[str, torch.Tensor],
        tau_target: torch.Tensor,
        torque_limit: float,
        lambda_physics: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        mse = F.mse_loss(outputs["tau"], tau_target)
        slack = torch.relu(outputs["tau"].abs() - torque_limit).mean()
        smooth = (outputs["tau"][:, 1:] - outputs["tau"][:, :-1]).pow(2).mean()
        consistency = slack + 0.1 * smooth
        total = mse + lambda_physics * consistency
        return total, {"mse": float(mse.item()), "consistency": float(consistency.item())}
