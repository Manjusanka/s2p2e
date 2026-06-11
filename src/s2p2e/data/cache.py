from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from s2p2e.data.corpus import ObjectRecord
from s2p2e.utils.geometry import load_objaverse_glb, load_shapenet_txt, normalize_point_cloud, sample_or_pad
from s2p2e.utils.io import ensure_dir


class PointCloudCache:
    def __init__(self, cache_root: str | Path, num_points: int) -> None:
        self.cache_root = ensure_dir(cache_root)
        self.num_points = num_points

    def _cache_path(self, record: ObjectRecord) -> Path:
        digest = hashlib.md5(f"{record.dataset}:{record.object_id}:{self.num_points}".encode("utf-8")).hexdigest()
        return self.cache_root / f"{digest}.npy"

    def load_or_build(self, record: ObjectRecord, seed: int) -> np.ndarray:
        path = self._cache_path(record)
        if path.exists():
            try:
                return np.load(path)
            except Exception:
                try:
                    path.unlink()
                except OSError:
                    pass
        rng = np.random.default_rng(seed)
        try:
            if record.dataset == "shapenet":
                points = load_shapenet_txt(record.path)
                points = sample_or_pad(points, self.num_points, rng)
            else:
                points = load_objaverse_glb(record.path, self.num_points, rng)
        except Exception:
            coords = rng.normal(0.0, 0.15, size=(self.num_points, 3)).astype(np.float32)
            normals = np.zeros((self.num_points, 3), dtype=np.float32)
            points = np.concatenate([coords, normals], axis=1)
        points = normalize_point_cloud(points).astype(np.float32)
        tmp_path = path.with_suffix(".tmp.npy")
        np.save(tmp_path, points)
        tmp_path.replace(path)
        return points
