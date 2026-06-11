from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


class PublicRobotTransitionDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, samples: list[dict[str, Any]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sample = self.samples[index]
        return {
            "obs": torch.tensor(sample["obs"], dtype=torch.float32),
            "action": torch.tensor(sample["action"], dtype=torch.float32),
            "next_obs": torch.tensor(sample["next_obs"], dtype=torch.float32),
            "reward": torch.tensor(sample["reward"], dtype=torch.float32),
            "physics_feat": torch.tensor(sample["physics_feat"], dtype=torch.float32),
            "predicted_tau": torch.tensor(sample["predicted_tau"], dtype=torch.float32),
        }


def load_droid_jsonl(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    samples: list[dict[str, Any]] = []
    for path in root.rglob("*.jsonl"):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                samples.append(json.loads(line))
    return samples


def load_rh20t_npz(root: str | Path) -> list[dict[str, Any]]:
    root = Path(root)
    samples: list[dict[str, Any]] = []
    for path in root.rglob("*.npz"):
        data = np.load(path)
        n = len(data["obs"])
        for i in range(n):
            samples.append(
                {
                    "obs": data["obs"][i].tolist(),
                    "action": data["action"][i].tolist(),
                    "next_obs": data["next_obs"][i].tolist(),
                    "reward": [float(data["reward"][i])],
                    "physics_feat": data["physics_feat"][i].tolist(),
                    "predicted_tau": data["predicted_tau"][i].tolist(),
                }
            )
    return samples


def build_synthetic_public_dataset(num_samples: int = 2048, obs_dim: int = 32, action_dim: int = 8, physics_dim: int = 512) -> list[dict[str, Any]]:
    rng = np.random.default_rng(42)
    samples = []
    for _ in range(num_samples):
        obs = rng.normal(size=(obs_dim,)).astype(np.float32)
        action = np.tanh(rng.normal(size=(action_dim,))).astype(np.float32)
        next_obs = (obs + 0.05 * rng.normal(size=(obs_dim,))).astype(np.float32)
        physics_feat = rng.normal(size=(physics_dim,)).astype(np.float32)
        predicted_tau = rng.normal(scale=0.5, size=(7,)).astype(np.float32)
        reward = np.array([1.0 - np.linalg.norm(action) * 0.05], dtype=np.float32)
        samples.append(
            {
                "obs": obs.tolist(),
                "action": action.tolist(),
                "next_obs": next_obs.tolist(),
                "reward": reward.tolist(),
                "physics_feat": physics_feat.tolist(),
                "predicted_tau": predicted_tau.tolist(),
            }
        )
    return samples
