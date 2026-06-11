from __future__ import annotations

from collections import defaultdict

import torch


def average_logs(logs: list[dict[str, float]]) -> dict[str, float]:
    merged: dict[str, list[float]] = defaultdict(list)
    for log in logs:
        for key, value in log.items():
            merged[key].append(float(value))
    return {key: sum(values) / max(len(values), 1) for key, values in merged.items()}


def move_to_device(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}
