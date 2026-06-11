from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scripts.convert_droid_subset import episode_to_transitions


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
