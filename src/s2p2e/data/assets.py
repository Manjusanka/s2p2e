from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from s2p2e.data.corpus import ObjectRecord
from s2p2e.utils.io import ensure_dir, save_json


@dataclass
class PhysicsAsset:
    object_id: str
    category: str
    friction: float
    restitution: float
    density: float
    stiffness: float
    damping: float


def synthesize_physics_assets(records: list[ObjectRecord], seed: int = 42) -> list[PhysicsAsset]:
    rng = np.random.default_rng(seed)
    assets: list[PhysicsAsset] = []
    for record in records:
        assets.append(
            PhysicsAsset(
                object_id=record.object_id,
                category=record.category,
                friction=float(rng.uniform(0.2, 1.1)),
                restitution=float(rng.uniform(0.0, 0.3)),
                density=float(rng.uniform(300.0, 3200.0)),
                stiffness=float(rng.uniform(50.0, 5000.0)),
                damping=float(rng.uniform(0.1, 8.0)),
            )
        )
    return assets


def export_physics_assets(path: str | Path, assets: list[PhysicsAsset]) -> None:
    path = ensure_dir(path)
    save_json(
        Path(path) / "physics_assets.json",
        {"assets": [asset.__dict__ for asset in assets]},
    )
