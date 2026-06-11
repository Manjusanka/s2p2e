from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.trainers.experiment_runner import run_experiment_suite


def main() -> None:
    payload = run_experiment_suite(ROOT / "configs" / "experiments.yaml")
    print(payload)


if __name__ == "__main__":
    main()
