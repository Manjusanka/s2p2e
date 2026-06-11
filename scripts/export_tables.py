from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]


def format_run(run_name: str) -> dict:
    metrics_dir = ROOT / "artifacts" / run_name / "metrics"
    summary_path = metrics_dir / "summary.json"
    if not summary_path.exists():
        return {"run_name": run_name, "status": "missing"}
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    return {
        "run_name": run_name,
        "joint_val_loss": summary.get("joint", {}).get("val", {}).get("loss"),
        "layer2_rmse": summary.get("layer2", {}).get("val", {}).get("rmse"),
        "num_records": summary.get("num_records"),
        "task_families": summary.get("task_families", {}),
    }


def main() -> None:
    rows = [format_run("s2p2e_small"), format_run("s2p2e_joint")]
    print({"rows": rows})


if __name__ == "__main__":
    main()
