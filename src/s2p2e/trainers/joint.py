from __future__ import annotations

import torch

from s2p2e.trainers.common import average_logs, move_to_device


def train_joint_epoch(model, loader, optimizer, device, cfg):
    model.train()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["points"],
            batch["tokens"],
            batch["task_id"],
            batch["geometric"],
            batch["operational"],
            batch["embodiment"],
            batch["qd_hist"],
            batch["tau_hist"],
            batch["pose_hist"],
            batch["force"],
            batch["pose_cur"],
            batch["phase"],
            batch["tau_pid"],
        )
        loss, metrics = model.joint_loss(
            outputs,
            batch["pose"],
            batch["tau_target"],
            cfg["lambda_joint_pose"],
            cfg["lambda_joint_tau"],
            cfg["lambda_joint_feasibility"],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logs.append({"loss": float(loss.item()), **metrics})
    return average_logs(logs)


@torch.no_grad()
def eval_joint_epoch(model, loader, device, cfg):
    model.eval()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["points"],
            batch["tokens"],
            batch["task_id"],
            batch["geometric"],
            batch["operational"],
            batch["embodiment"],
            batch["qd_hist"],
            batch["tau_hist"],
            batch["pose_hist"],
            batch["force"],
            batch["pose_cur"],
            batch["phase"],
            batch["tau_pid"],
        )
        loss, metrics = model.joint_loss(
            outputs,
            batch["pose"],
            batch["tau_target"],
            cfg["lambda_joint_pose"],
            cfg["lambda_joint_tau"],
            cfg["lambda_joint_feasibility"],
        )
        logs.append({"loss": float(loss.item()), **metrics})
    return average_logs(logs)
