from __future__ import annotations

from collections import defaultdict

import torch

from s2p2e.trainers.common import move_to_device


@torch.no_grad()
def evaluate_task_families(model, loader, device, cfg, task_names):
    model.eval()
    by_task = defaultdict(list)
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
        losses = ((outputs["layer1"]["poses"][:, 0] - batch["pose"]) ** 2).mean(dim=-1)
        for idx, task_id in enumerate(batch["task_id"].tolist()):
            by_task[task_names[task_id]].append(float(losses[idx].item()))
    return {task: {"mean_loss": sum(vals) / len(vals), "count": len(vals)} for task, vals in by_task.items()}


@torch.no_grad()
def evaluate_kb_tiers(layer1_model, loader, device, cfg):
    layer1_model.eval()
    settings = {
        "full_3tier": (True, True, True),
        "semantic_only": (True, False, False),
        "semantic_geometric": (True, True, False),
        "semantic_operational": (True, False, True),
    }
    results = {}
    for name, flags in settings.items():
        layer1_model.set_kb_tiers(*flags)
        losses = []
        for batch in loader:
            batch = move_to_device(batch, device)
            outputs = layer1_model(batch["points"], batch["tokens"], batch["geometric"], batch["operational"])
            loss, _ = layer1_model.loss(
                outputs,
                batch["pose"],
                cfg["lambda_pose"],
                cfg["lambda_gripper"],
                cfg["lambda_retrieval"],
            )
            losses.append(float(loss.item()))
        results[name] = {"loss": sum(losses) / len(losses)}
    layer1_model.set_kb_tiers(True, True, True)
    return results
