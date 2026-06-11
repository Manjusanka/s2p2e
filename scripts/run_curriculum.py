from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.trainers.curriculum import build_curriculum_phases
from s2p2e.trainers.pipeline import run_pipeline
from s2p2e.utils.config import load_config


def main() -> None:
    config = load_config(ROOT / "configs" / "experiment_joint.yaml")
    phases = build_curriculum_phases(config)
    results = []
    for idx, phase_cfg in enumerate(phases, start=1):
        phase_cfg["run_name"] = f"{config['run_name']}_{idx}_{phase_cfg['phase_name']}"
        result = run_pipeline(phase_cfg)
        results.append({"phase": phase_cfg["phase_name"], "summary": result})
    print({"curriculum_results": results})


if __name__ == "__main__":
    main()
