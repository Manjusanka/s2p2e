from __future__ import annotations

import torch


def pose_metrics(pred_pose: torch.Tensor, target_pose: torch.Tensor) -> dict[str, float]:
    return {
        "position_mae": float((pred_pose[:, :3] - target_pose[:, :3]).abs().mean().item()),
        "quat_mae": float((pred_pose[:, 3:7] - target_pose[:, 3:7]).abs().mean().item()),
        "gripper_mae": float((pred_pose[:, 7] - target_pose[:, 7]).abs().mean().item()),
    }


def physics_metrics(pred_tau: torch.Tensor, target_tau: torch.Tensor, torque_limit: float) -> dict[str, float]:
    violations = (pred_tau.abs() > torque_limit).float().mean().item()
    rmse = torch.sqrt(torch.mean((pred_tau - target_tau) ** 2)).item()
    return {"torque_violation_rate": float(violations), "rmse": float(rmse)}


def control_metrics(command: torch.Tensor, target_residual: torch.Tensor) -> dict[str, float]:
    return {
        "command_norm": float(command.norm(dim=-1).mean().item()),
        "residual_rmse": float(torch.sqrt(torch.mean((command - target_residual) ** 2)).item()),
    }
