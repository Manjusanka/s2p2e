from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from s2p2e.data.synthetic import embodiment_vector, synthetic_inverse_dynamics


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


def build_cross_embodiment_public_dataset(num_samples: int = 8192, seed: int = 42) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    samples: list[dict[str, Any]] = []
    base_embodiment = embodiment_vector(7)
    for idx in range(num_samples):
        scale = np.array(
            [
                rng.uniform(0.75, 1.25),
                rng.uniform(0.65, 1.35),
                rng.uniform(0.70, 1.30),
            ],
            dtype=np.float32,
        )
        embodiment = base_embodiment.copy()
        embodiment[:7] *= scale[0]
        embodiment[7:14] *= scale[1]
        embodiment[14:21] *= scale[2]
        load_mass = float(rng.uniform(0.2, 3.0))
        q = rng.uniform(-1.0, 1.0, size=(7,)).astype(np.float32)
        qd = rng.normal(scale=0.25, size=(7,)).astype(np.float32)
        qdd = rng.normal(scale=0.15, size=(7,)).astype(np.float32)
        tau = synthetic_inverse_dynamics(q, qd, qdd, embodiment, load_mass=load_mass)
        cart = np.array(
            [
                np.sin(q[0]),
                np.cos(q[1]),
                np.sin(q[2]) * 0.5,
                q[3],
                q[4],
                q[5],
            ],
            dtype=np.float32,
        )
        gripper = np.array([rng.uniform(0.0, 1.0)], dtype=np.float32)
        obs = np.concatenate([cart, gripper, q, qd], axis=0).astype(np.float32)
        action = np.tanh(tau / 12.0 + rng.normal(scale=0.03, size=(7,))).astype(np.float32)
        next_q = q + 0.05 * qd + 0.01 * qdd
        next_qd = qd + 0.05 * qdd
        next_cart = np.array(
            [
                np.sin(next_q[0]),
                np.cos(next_q[1]),
                np.sin(next_q[2]) * 0.5,
                next_q[3],
                next_q[4],
                next_q[5],
            ],
            dtype=np.float32,
        )
        next_obs = np.concatenate([next_cart, gripper, next_q, next_qd], axis=0).astype(np.float32)
        physics_feat = np.concatenate([q, qd, tau, embodiment, np.array([load_mass], dtype=np.float32)], axis=0)
        physics_feat = np.pad(physics_feat, (0, max(0, 64 - physics_feat.shape[0])))[:64].astype(np.float32)
        torque_penalty = np.maximum(np.abs(tau) - 12.0, 0.0).mean()
        smoothness = np.linalg.norm(action) / np.sqrt(action.size)
        reward = np.array([1.0 - 0.08 * smoothness - 0.2 * torque_penalty], dtype=np.float32)
        samples.append(
            {
                "obs": obs.tolist(),
                "action": action.tolist(),
                "next_obs": next_obs.tolist(),
                "reward": reward.tolist(),
                "physics_feat": physics_feat.tolist(),
                "predicted_tau": tau.astype(np.float32).tolist(),
                "source": "synthetic_cross_embodiment",
                "embodiment_id": f"synthetic_embodiment_{idx % 16:02d}",
            }
        )
    return samples
