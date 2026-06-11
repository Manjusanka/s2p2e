from __future__ import annotations

from pathlib import Path
import json
import sys

import numpy as np
from tfrecord.reader import tfrecord_loader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.synthetic import embodiment_vector, synthetic_inverse_dynamics
from s2p2e.utils.io import ensure_dir, save_json


def reshape_sequence(flat, dims):
    arr = np.asarray(flat, dtype=np.float32)
    if not dims:
        return arr
    steps = arr.size // int(np.prod(dims))
    return arr.reshape(steps, *dims)


def episode_to_transitions(example: dict, limit_steps: int | None = None) -> list[dict]:
    action = reshape_sequence(example["steps/action"], (7,))
    joint_pos = reshape_sequence(example["steps/action_dict/joint_position"], (7,))
    joint_vel = reshape_sequence(example["steps/action_dict/joint_velocity"], (7,))
    cart = reshape_sequence(example["steps/observation/cartesian_position"], (6,))
    gripper = reshape_sequence(example["steps/observation/gripper_position"], (1,))
    reward = reshape_sequence(example["steps/reward"], ())

    steps = min(len(reward), len(action), len(joint_pos), len(joint_vel), len(cart), len(gripper))
    if limit_steps is not None:
        steps = min(steps, limit_steps)
    embodiment = embodiment_vector(7)
    transitions = []
    for i in range(max(steps - 1, 0)):
        obs = np.concatenate([cart[i], gripper[i], joint_pos[i], joint_vel[i]], axis=0).astype(np.float32)
        next_obs = np.concatenate([cart[i + 1], gripper[i + 1], joint_pos[i + 1], joint_vel[i + 1]], axis=0).astype(np.float32)
        q = joint_pos[i]
        qd = joint_vel[i]
        qdd = np.zeros_like(qd)
        predicted_tau = synthetic_inverse_dynamics(q, qd, qdd, embodiment, load_mass=1.0)
        physics_feat = np.concatenate(
            [q, qd, predicted_tau, embodiment[:7], embodiment[7:14], embodiment[14:21]],
            axis=0,
        ).astype(np.float32)
        if physics_feat.shape[0] < 64:
            physics_feat = np.pad(physics_feat, (0, 64 - physics_feat.shape[0]))
        elif physics_feat.shape[0] > 64:
            physics_feat = physics_feat[:64]
        transitions.append(
            {
                "obs": obs.tolist(),
                "action": action[i].astype(np.float32).tolist(),
                "next_obs": next_obs.tolist(),
                "reward": [float(reward[i])],
                "physics_feat": physics_feat.tolist(),
                "predicted_tau": predicted_tau.astype(np.float32).tolist(),
            }
        )
    return transitions


def main() -> None:
    source_dir = ROOT / "public_data" / "droid_100" / "1.0.0"
    out_dir = ensure_dir(ROOT / "public_data" / "droid_subset_jsonl")
    all_transitions = []
    shard_paths = sorted(source_dir.glob("r2d2_faceblur-train.tfrecord-*"))
    for shard_path in shard_paths:
        loader = tfrecord_loader(str(shard_path), None)
        shard_out = out_dir / f"{shard_path.name}.jsonl"
        count = 0
        with shard_out.open("w", encoding="utf-8") as handle:
            for example in loader:
                transitions = episode_to_transitions(example, limit_steps=64)
                for item in transitions:
                    handle.write(json.dumps(item) + "\n")
                all_transitions.extend(transitions)
                count += len(transitions)
        print({"shard": shard_path.name, "transitions": count})
    save_json(out_dir / "manifest.json", {"num_transitions": len(all_transitions), "shards": len(shard_paths)})
    print({"output_dir": str(out_dir), "num_transitions": len(all_transitions)})


if __name__ == "__main__":
    main()
