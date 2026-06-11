from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from s2p2e.data.corpus import ObjectRecord
from s2p2e.data.cache import PointCloudCache
from s2p2e.data.synthetic import (
    build_geometry_targets,
    build_prompt,
    embodiment_vector,
    synthetic_inverse_dynamics,
)
from s2p2e.utils.geometry import (
    load_objaverse_glb,
    load_shapenet_txt,
    normalize_point_cloud,
    sample_or_pad,
)
from s2p2e.utils.text import SimpleTokenizer


@dataclass
class Layer1Sample:
    points: torch.Tensor
    tokens: torch.Tensor
    task_id: torch.Tensor
    pose: torch.Tensor
    geometric: torch.Tensor
    operational: torch.Tensor


class Layer1Dataset(Dataset[Layer1Sample]):
    def __init__(
        self,
        records: list[ObjectRecord],
        tasks: list[str],
        tokenizer: SimpleTokenizer,
        num_points: int,
        split: str,
        cache: PointCloudCache | None = None,
    ) -> None:
        self.records = records
        self.tasks = tasks
        self.tokenizer = tokenizer
        self.num_points = num_points
        self.split = split
        self.cache = cache

    def __len__(self) -> int:
        return len(self.records) * len(self.tasks)

    def _load_points(self, record: ObjectRecord, seed: int) -> np.ndarray:
        if self.cache is not None:
            return self.cache.load_or_build(record, seed)
        rng = np.random.default_rng(seed)
        if record.dataset == "shapenet":
            points = load_shapenet_txt(record.path)
            points = sample_or_pad(points, self.num_points, rng)
        else:
            points = load_objaverse_glb(record.path, self.num_points, rng)
        return normalize_point_cloud(points).astype(np.float32)

    def __getitem__(self, index: int) -> Layer1Sample:
        record = self.records[index // len(self.tasks)]
        task_id = index % len(self.tasks)
        task = self.tasks[task_id]
        points = self._load_points(record, seed=index + 17)
        targets = build_geometry_targets(points, task_id)
        text = f"{build_prompt(record.category, task)} {record.affordance_text}"
        return Layer1Sample(
            points=torch.from_numpy(points),
            tokens=self.tokenizer.encode(text),
            task_id=torch.tensor(task_id, dtype=torch.long),
            pose=torch.from_numpy(targets.pose),
            geometric=torch.from_numpy(targets.geometric),
            operational=torch.from_numpy(targets.operational),
        )


class PhysicsTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: list[ObjectRecord], num_samples: int, trajectory_length: int, dof: int) -> None:
        self.records = records
        self.num_samples = num_samples
        self.trajectory_length = trajectory_length
        self.dof = dof
        self.embodiment = embodiment_vector(dof)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(index + 101)
        steps = self.trajectory_length
        t = np.linspace(0.0, 1.0, steps, dtype=np.float32)
        phase = rng.uniform(0.1, 1.0, size=(self.dof,)).astype(np.float32)
        q = np.stack([0.6 * np.sin(2.0 * math.pi * (t + p)) for p in phase], axis=1)
        qd = np.gradient(q, axis=0).astype(np.float32)
        qdd = np.gradient(qd, axis=0).astype(np.float32)
        load_mass = float(0.5 + (index % max(len(self.records), 1)) * 0.01)
        tau = np.stack(
            [synthetic_inverse_dynamics(q[i], qd[i], qdd[i], self.embodiment, load_mass) for i in range(steps)],
            axis=0,
        )
        return {
            "state": torch.from_numpy(np.concatenate([q, qd, qdd], axis=1)),
            "tau": torch.from_numpy(tau),
            "embodiment": torch.from_numpy(self.embodiment),
        }


class ResidualSignalDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, num_samples: int, history_length: int, dof: int) -> None:
        self.num_samples = num_samples
        self.history_length = history_length
        self.dof = dof

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        rng = np.random.default_rng(index + 303)
        h = self.history_length
        qd_hist = rng.normal(0.0, 0.4, size=(h, self.dof)).astype(np.float32)
        tau_hist = rng.normal(0.0, 0.3, size=(h, self.dof)).astype(np.float32)
        pose_hist = rng.normal(0.0, 0.2, size=(h, 7)).astype(np.float32)
        force = rng.normal(0.0, 1.0, size=(6,)).astype(np.float32)
        pose_cur = pose_hist[-1]
        phase = int(index % 5)
        phase_onehot = np.eye(5, dtype=np.float32)[phase]

        tau_fric = 0.2 * qd_hist[-1] + 0.1 * tau_hist.mean(axis=0)
        tau_load = 0.05 * force.mean() + 0.04 * pose_cur[: self.dof]
        tau_noise = 0.08 * pose_hist.std(axis=0)[: self.dof]
        residual = tau_fric + tau_load + tau_noise
        alpha_target = np.array([0.15 if phase < 2 else 0.85], dtype=np.float32)

        return {
            "qd_hist": torch.from_numpy(qd_hist),
            "tau_hist": torch.from_numpy(tau_hist),
            "pose_hist": torch.from_numpy(pose_hist),
            "force": torch.from_numpy(force),
            "pose_cur": torch.from_numpy(pose_cur),
            "phase": torch.from_numpy(phase_onehot),
            "target_residual": torch.from_numpy(residual.astype(np.float32)),
            "target_alpha": torch.from_numpy(alpha_target),
            "tau_pid": torch.from_numpy(rng.normal(1.0, 0.3, size=(self.dof,)).astype(np.float32)),
        }


class JointStackDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(
        self,
        records: list[ObjectRecord],
        tasks: list[str],
        tokenizer: SimpleTokenizer,
        num_points: int,
        history_length: int,
        dof: int,
        cache: PointCloudCache | None = None,
    ) -> None:
        self.layer1 = Layer1Dataset(records, tasks, tokenizer, num_points, "joint", cache=cache)
        self.history_length = history_length
        self.dof = dof

    def __len__(self) -> int:
        return len(self.layer1)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        base = self.layer1[index]
        rng = np.random.default_rng(index + 707)
        qd_hist = rng.normal(0.0, 0.4, size=(self.history_length, self.dof)).astype(np.float32)
        tau_hist = rng.normal(0.0, 0.3, size=(self.history_length, self.dof)).astype(np.float32)
        pose_hist = rng.normal(0.0, 0.2, size=(self.history_length, 7)).astype(np.float32)
        embodiment = embodiment_vector(self.dof)
        q_ref = rng.normal(0.0, 0.5, size=(self.dof,)).astype(np.float32)
        qd_ref = np.zeros((self.dof,), dtype=np.float32)
        qdd_ref = np.zeros((self.dof,), dtype=np.float32)
        tau_target = synthetic_inverse_dynamics(q_ref, qd_ref, qdd_ref, embodiment, load_mass=1.0)
        phase = np.eye(5, dtype=np.float32)[index % 5]
        tau_pid = np.abs(rng.normal(1.0, 0.2, size=(self.dof,)).astype(np.float32)) + 0.1
        return {
            "points": base.points,
            "tokens": base.tokens,
            "task_id": base.task_id,
            "pose": base.pose,
            "geometric": base.geometric,
            "operational": base.operational,
            "embodiment": torch.from_numpy(embodiment),
            "qd_hist": torch.from_numpy(qd_hist),
            "tau_hist": torch.from_numpy(tau_hist),
            "pose_hist": torch.from_numpy(pose_hist),
            "force": torch.from_numpy(rng.normal(0.0, 1.0, size=(6,)).astype(np.float32)),
            "pose_cur": torch.from_numpy(pose_hist[-1]),
            "phase": torch.from_numpy(phase),
            "tau_pid": torch.from_numpy(tau_pid),
            "tau_target": torch.from_numpy(tau_target),
        }


def collate_layer1(batch: list[Layer1Sample]) -> dict[str, torch.Tensor]:
    return {
        "points": torch.stack([sample.points for sample in batch], dim=0),
        "tokens": torch.stack([sample.tokens for sample in batch], dim=0),
        "task_id": torch.stack([sample.task_id for sample in batch], dim=0),
        "pose": torch.stack([sample.pose for sample in batch], dim=0),
        "geometric": torch.stack([sample.geometric for sample in batch], dim=0),
        "operational": torch.stack([sample.operational for sample in batch], dim=0),
    }


def collate_joint(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    keys = batch[0].keys()
    return {key: torch.stack([sample[key] for sample in batch], dim=0) for key in keys}


def split_records(records: list[ObjectRecord], train_ratio: float, val_ratio: float) -> dict[str, list[ObjectRecord]]:
    total = len(records)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": records[:train_end],
        "val": records[train_end:val_end],
        "test": records[val_end:],
    }


def build_tokenizer(records: list[ObjectRecord], tasks: list[str]) -> SimpleTokenizer:
    texts = []
    for record in records:
        for task in tasks:
            texts.append(f"{build_prompt(record.category, task)} {record.affordance_text}")
    return SimpleTokenizer(texts)
