from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def main() -> None:
    rag_root = ROOT / "artifacts" / "s2p2e_rag_full"
    physics_root = ROOT / "artifacts" / "s2p2e_physics_full"
    policy_root = ROOT / "artifacts" / "actor_critic_droid_full"
    out_dir = ROOT / "public_data" / "s2p2e_rag_dataset"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = json.loads((rag_root / "kb" / "kb_entries.json").read_text(encoding="utf-8"))["entries"]
    rag_summary = json.loads((rag_root / "metrics" / "rag_layer1_summary.json").read_text(encoding="utf-8"))
    physics_summary = json.loads((physics_root / "metrics" / "summary.json").read_text(encoding="utf-8"))
    policy_summary = json.loads((policy_root / "actor_critic_droid_history.json").read_text(encoding="utf-8"))

    with (out_dir / "entries.jsonl").open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    train_end = int(len(entries) * 0.8)
    val_end = train_end + int(len(entries) * 0.1)
    splits = {"train": entries[:train_end], "val": entries[train_end:val_end], "test": entries[val_end:]}
    for split, split_entries in splits.items():
        with (out_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for entry in split_entries:
                handle.write(json.dumps(entry, ensure_ascii=True) + "\n")

    category_counts = Counter(entry["category"] for entry in entries)
    task_counts = Counter(entry["task"] for entry in entries)
    dataset_counts = Counter(entry["dataset"] for entry in entries)
    manifest = {
        "name": "S2P2E-RAG",
        "version": "2026-06-11",
        "num_entries": len(entries),
        "num_objects": rag_summary["records"],
        "num_categories": len(category_counts),
        "tasks": sorted(task_counts),
        "datasets": dict(sorted(dataset_counts.items())),
        "splits": {name: len(value) for name, value in splits.items()},
        "fields": ["object_id", "dataset", "category", "task", "semantic_text", "geometric", "operational"],
        "source_artifacts": {
            "kb_entries": "artifacts/s2p2e_rag_full/kb/kb_entries.json",
            "rag_checkpoint": "artifacts/s2p2e_rag_full/checkpoints/layer1_best.pt",
            "full_stack_checkpoint": "artifacts/s2p2e_physics_full/checkpoints/joint_best.pt",
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    write_json(out_dir / "category_counts.json", dict(sorted(category_counts.items())))
    write_json(out_dir / "task_counts.json", dict(sorted(task_counts.items())))

    card = f"""# S2P2E-RAG Dataset

S2P2E-RAG is the retrieval knowledge-base dataset generated for the S2P2E manipulation stack. It pairs object-level 3D assets with five tabletop manipulation task families and stores three retrieval tiers for each entry: semantic text, geometric descriptors, and operational pose/action descriptors.

## Scope

- Objects: {rag_summary['records']}
- Retrieval entries: {len(entries)}
- Categories: {len(category_counts)}
- Tasks: {', '.join(sorted(task_counts))}
- Splits: train {len(splits['train'])}, val {len(splits['val'])}, test {len(splits['test'])}

## Intended Use

This dataset is intended for retrieval-augmented pose proposal, manipulation knowledge-base ablations, and reproducibility of the Layer I S2P2E experiments. It is not a full replacement for upstream Objaverse, ShapeNet/ModelNet, or DROID assets.

## Training Snapshot

- RAG best epoch: {rag_summary['best']['epoch']}
- RAG validation pose MAE: {rag_summary['best']['val']['pose_mae']:.6f}
- Full-stack validation loss: {physics_summary['joint']['val']['loss']:.6f}
- DROID policy samples: {policy_summary['num_samples']}

## Files

- `entries.jsonl`: all retrieval entries.
- `train.jsonl`, `val.jsonl`, `test.jsonl`: deterministic split files.
- `manifest.json`: dataset metadata.
- `category_counts.json`, `task_counts.json`: dataset statistics.
"""
    (out_dir / "DATASET_CARD.md").write_text(card, encoding="utf-8")

    csv_path = ROOT / "paper" / "source_data" / "training_expansion_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["module", "training_data", "best_epoch", "primary_validation_metric", "value"])
        writer.writerow(["Layer I / S2P2E-RAG", f"{rag_summary['records']} objects; {len(entries)} KB entries", rag_summary["best"]["epoch"], "pose_mae", rag_summary["best"]["val"]["pose_mae"]])
        writer.writerow(["Layer II / PPE", "4096 synthetic inverse-dynamics trajectories", physics_summary["layer2"]["epoch"], "rmse", physics_summary["layer2"]["val"]["rmse"]])
        writer.writerow(["Layer III / residual control", "4096 residual-control trajectories", physics_summary["layer3"]["epoch"], "residual_loss", physics_summary["layer3"]["val"]["residual_loss"]])
        writer.writerow(["Joint S2P2E", f"{physics_summary['num_records']} 3D objects x 5 tasks", physics_summary["joint"]["epoch"], "joint_val_loss", physics_summary["joint"]["val"]["loss"]])
        writer.writerow(["Actor-Critic DROID bridge", f"{policy_summary['num_samples']} DROID transitions", policy_summary["history"][-1]["epoch"], "critic_loss", policy_summary["history"][-1]["critic_loss"]])

    print({"dataset_dir": str(out_dir), "entries": len(entries), "source_data": str(csv_path)})


if __name__ == "__main__":
    main()
