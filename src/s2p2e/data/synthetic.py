from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from s2p2e.utils.geometry import principal_axes, synthetic_pose_from_points


TASK_TEMPLATES = {
    "pick": "pick the {category}",
    "place": "place the {category} at the target zone",
    "insert": "insert the {category} into the slot",
    "pour": "pour using the {category}",
    "tool_use": "use the {category} as a tool",
}


@dataclass
class GeometryTargets:
    geometric: np.ndarray
    operational: np.ndarray
    pose: np.ndarray
    mass: float


def build_prompt(category: str, task: str) -> str:
    return TASK_TEMPLATES[task].format(category=category.replace("_", " "))


def build_geometry_targets(points: np.ndarray, task_index: int) -> GeometryTargets:
    coords = points[:, :3]
    center = coords.mean(axis=0)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    extents = maxs - mins
    eigvals, _ = principal_axes(points)
    curvature = eigvals / (eigvals.sum() + 1e-6)
    approachability = np.array(
        [extents[2], extents[0] / (extents[1] + 1e-6), float(coords[:, 2].max())],
        dtype=np.float32,
    )
    geometric = np.concatenate([center, extents, curvature.astype(np.float32), approachability], axis=0)
    pose = synthetic_pose_from_points(points, task_index)
    grasp_quality = np.array(
        [
            float(1.0 / (1.0 + np.linalg.norm(extents))),
            float(curvature.max()),
            float(extents.mean()),
            float(task_index),
        ],
        dtype=np.float32,
    )
    operational = np.concatenate([pose, grasp_quality], axis=0)
    mass = float(np.clip(extents.prod() * 8.0, 0.2, 4.0))
    return GeometryTargets(geometric=geometric.astype(np.float32), operational=operational, pose=pose, mass=mass)


def embodiment_vector(dof: int = 7) -> np.ndarray:
    links = np.linspace(0.22, 0.08, dof, dtype=np.float32)
    masses = np.linspace(4.0, 1.0, dof, dtype=np.float32)
    inertias = np.linspace(0.08, 0.01, dof, dtype=np.float32)
    return np.concatenate([links, masses, inertias], axis=0)


def synthetic_inverse_dynamics(
    q: np.ndarray,
    qd: np.ndarray,
    qdd: np.ndarray,
    embodiment: np.ndarray,
    load_mass: float,
) -> np.ndarray:
    dof = q.shape[-1]
    links = embodiment[:dof]
    masses = embodiment[dof : 2 * dof]
    inertias = embodiment[2 * dof : 3 * dof]
    gravity = np.sin(q) * links * masses * 0.4
    inertia_term = qdd * inertias * 6.0
    damping = qd * (0.08 + 0.02 * np.arange(1, dof + 1, dtype=np.float32))
    load = load_mass * 0.05 * np.cos(q + np.linspace(0.0, 0.5, dof, dtype=np.float32))
    return (gravity + inertia_term + damping + load).astype(np.float32)
