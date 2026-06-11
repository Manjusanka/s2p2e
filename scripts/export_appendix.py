from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    appendix = ROOT / "artifacts" / "s2p2e_joint" / "metrics" / "method_appendix.json"
    if not appendix.exists():
        print({"missing": str(appendix)})
        return
    with appendix.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    print(payload)


if __name__ == "__main__":
    main()
