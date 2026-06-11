from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.trainers.end_to_end import run_full_pipeline


def main() -> None:
    summary = run_full_pipeline(ROOT / "configs" / "full_pipeline.yaml")
    print(summary)


if __name__ == "__main__":
    main()
