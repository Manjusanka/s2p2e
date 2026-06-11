from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import trimesh


def sample_or_pad(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    count = points.shape[0]
    if count >= num_points:
        idx = rng.choice(count, size=num_points, replace=False)
        return points[idx]
    idx = rng.choice(count, size=num_points - count, replace=True)
    return np.concatenate([points, points[idx]], axis=0)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    coords = points[:, :3]
    center = coords.mean(axis=0, keepdims=True)
    coords = coords - center
    scale = np.linalg.norm(coords, axis=1).max()
    scale = float(scale) if scale > 1e-8 else 1.0
    coords = coords / scale
    if points.shape[1] > 3:
        return np.concatenate([coords, points[:, 3:]], axis=1)
    return coords


def load_shapenet_txt(path: str | Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",", dtype=np.float32)
    if data.ndim == 1:
        data = data[None, :]
    if data.shape[1] < 3:
        raise ValueError(f"Point cloud at {path} has fewer than 3 columns")
    if data.shape[1] == 3:
        normals = np.zeros((data.shape[0], 3), dtype=np.float32)
        data = np.concatenate([data, normals], axis=1)
    return data[:, :6]


def load_objaverse_glb(path: str | Path, num_points: int, rng: np.random.Generator) -> np.ndarray:
    mesh = trimesh.load(path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        geometries = [g for g in mesh.geometry.values() if isinstance(g, trimesh.Trimesh)]
        mesh = trimesh.util.concatenate(geometries) if geometries else trimesh.Trimesh()
    if getattr(mesh, "faces", None) is not None and len(mesh.faces) > 0:
        samples, face_idx = trimesh.sample.sample_surface(mesh, num_points)
        normals = mesh.face_normals[face_idx]
        return np.concatenate([samples.astype(np.float32), normals.astype(np.float32)], axis=1)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    if vertices.size == 0:
        raise ValueError(f"Objaverse mesh has no valid geometry: {path}")
    sampled = sample_or_pad(vertices, num_points, rng)
    normals = np.zeros((sampled.shape[0], 3), dtype=np.float32)
    return np.concatenate([sampled, normals], axis=1)


def principal_axes(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coords = points[:, :3]
    cov = np.cov(coords.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return eigvals[order], eigvecs[:, order]


def rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    m = matrix
    trace = np.trace(m)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2.0
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2.0
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2.0
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qw, qx, qy, qz], dtype=np.float32)
    norm = np.linalg.norm(quat)
    return quat / (norm if norm > 1e-8 else 1.0)


def synthetic_pose_from_points(points: np.ndarray, task_index: int) -> np.ndarray:
    coords = points[:, :3]
    center = coords.mean(axis=0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = maxs - mins
    _, axes = principal_axes(points)
    rotation = np.stack([axes[:, 0], axes[:, 1], np.cross(axes[:, 0], axes[:, 1])], axis=1)
    quat = rotation_matrix_to_quaternion(rotation)
    offsets = np.array(
        [
            [0.0, 0.0, extents[2] * 0.2],
            [0.0, 0.0, extents[2] * 0.4],
            [0.0, extents[1] * 0.1, extents[2] * 0.3],
            [0.0, -extents[1] * 0.1, extents[2] * 0.45],
            [extents[0] * 0.15, 0.0, extents[2] * 0.25],
        ],
        dtype=np.float32,
    )
    position = center + offsets[task_index % len(offsets)]
    gripper = np.array([1.0 if float(extents.mean()) > 0.35 else 0.0], dtype=np.float32)
    return np.concatenate([position.astype(np.float32), quat, gripper], axis=0)
