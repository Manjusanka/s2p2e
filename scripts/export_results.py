from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    summary_path = ROOT / "artifacts" / "s2p2e_joint" / "metrics" / "summary.json"
    if not summary_path.exists():
        print({"missing": str(summary_path)})
        return
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(payload)


if __name__ == "__main__":
    main()
