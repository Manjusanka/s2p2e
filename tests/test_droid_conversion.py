from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.convert_droid_subset import episode_to_transitions
from s2p2e.data.public_datasets import build_cross_embodiment_public_dataset


def test_episode_to_transitions_shapes() -> None:
    example = {
        "steps/action": [0.0] * (4 * 7),
        "steps/action_dict/joint_position": [0.0] * (4 * 7),
        "steps/action_dict/joint_velocity": [0.0] * (4 * 7),
        "steps/observation/cartesian_position": [0.0] * (4 * 6),
        "steps/observation/gripper_position": [0.0] * 4,
        "steps/reward": [0.0] * 4,
    }
    transitions = episode_to_transitions(example)
    assert len(transitions) == 3
    assert len(transitions[0]["obs"]) == 21
    assert len(transitions[0]["action"]) == 7


def test_cross_embodiment_samples_match_droid_shapes() -> None:
    samples = build_cross_embodiment_public_dataset(4)
    sample = samples[0]

    assert len(samples) == 4
    assert len(sample["obs"]) == 21
    assert len(sample["action"]) == 7
    assert len(sample["next_obs"]) == 21
    assert len(sample["physics_feat"]) == 64
    assert len(sample["predicted_tau"]) == 7
    assert sample["source"] == "synthetic_cross_embodiment"
