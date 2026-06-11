from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.trainers.curriculum import build_curriculum_phases
from s2p2e.trainers.experiment_runner import run_single_experiment
from s2p2e.utils.config import load_config


def test_curriculum_builder() -> None:
    cfg = load_config(ROOT / "configs" / "experiment_joint.yaml")
    phases = build_curriculum_phases(cfg)
    assert len(phases) == 3
    assert phases[0]["data"]["tasks"] == ["pick", "place"]


def test_run_single_experiment_smoke() -> None:
    result = run_single_experiment(ROOT / "configs" / "experiment_small.yaml")
    assert "summary" in result
