from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class FrictionNet(nn.Module):
    def __init__(self, dof: int, history_length: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(dof * 2, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.head = nn.Linear(64 * history_length, dof)

    def forward(self, qd_hist: torch.Tensor, tau_hist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([qd_hist, tau_hist], dim=-1).transpose(1, 2)
        x = self.net(x).flatten(1)
        return self.head(x)


class LoadNet(nn.Module):
    def __init__(self, dof: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(13, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, dof),
        )

    def forward(self, force: torch.Tensor, pose_cur: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([force, pose_cur], dim=-1))


class NoiseNet(nn.Module):
    def __init__(self, dof: int) -> None:
        super().__init__()
        self.gru = nn.GRU(7, 64, batch_first=True)
        self.head = nn.Linear(64, dof)

    def forward(self, pose_hist: torch.Tensor) -> torch.Tensor:
        _, hidden = self.gru(pose_hist)
        return self.head(hidden[-1])


class GateNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, phase: torch.Tensor, force: torch.Tensor) -> torch.Tensor:
        magnitude = force.norm(dim=-1, keepdim=True)
        return torch.sigmoid(self.net(torch.cat([phase, magnitude], dim=-1)))


class ResidualController(nn.Module):
    def __init__(self, dof: int, history_length: int, clip_ratio: float = 0.2) -> None:
        super().__init__()
        self.friction = FrictionNet(dof, history_length)
        self.load = LoadNet(dof)
        self.noise = NoiseNet(dof)
        self.gate = GateNet()
        self.clip_ratio = clip_ratio

    def forward(
        self,
        qd_hist: torch.Tensor,
        tau_hist: torch.Tensor,
        pose_hist: torch.Tensor,
        force: torch.Tensor,
        pose_cur: torch.Tensor,
        phase: torch.Tensor,
        tau_pid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        tau_fric = self.friction(qd_hist, tau_hist)
        tau_load = self.load(force, pose_cur)
        tau_noise = self.noise(pose_hist)
        residual = tau_fric + tau_load + tau_noise
        alpha = self.gate(phase, force)
        bound = self.clip_ratio * tau_pid.abs()
        clipped = torch.clamp(residual, -bound, bound)
        command = tau_pid + alpha * clipped
        return {"residual": residual, "alpha": alpha, "clipped": clipped, "command": command}

    @staticmethod
    def loss(
        outputs: dict[str, torch.Tensor],
        target_residual: torch.Tensor,
        target_alpha: torch.Tensor,
        lambda_gate: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        residual_loss = F.mse_loss(outputs["clipped"], target_residual)
        gate_loss = F.mse_loss(outputs["alpha"], target_alpha)
        total = residual_loss + lambda_gate * gate_loss
        return total, {"residual_loss": float(residual_loss.item()), "gate_loss": float(gate_loss.item())}
