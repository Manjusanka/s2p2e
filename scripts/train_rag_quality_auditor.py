from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.models.rag_quality import RAGQualityAuditor
from s2p2e.utils.config import load_config
from s2p2e.utils.io import ensure_dir, save_checkpoint, save_json
from s2p2e.utils.seed import set_seed


TASKS = ["pick", "place", "insert", "pour", "tool_use"]
DATASETS = ["objaverse", "shapenet"]
CONTAINER_CATEGORIES = {"bottle", "bowl", "cup", "glass_box", "vase"}
TOOL_LIKE_CATEGORIES = {"guitar", "keyboard", "lamp", "laptop", "monitor", "xbox"}


def clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def one_hot(value: str, choices: list[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def feature_vector(entry: dict[str, Any]) -> list[float]:
    geometric = [float(x) for x in entry["geometric"]]
    operational = [float(x) for x in entry["operational"]]
    geometry = entry["geometry_summary"]
    operational_summary = entry["operational_summary"]
    physics = entry["physics_prior"]
    flags = entry["quality_flags"]
    scalar = [
        float(geometry["volume_proxy"]),
        float(geometry["aspect_ratio"]),
        float(geometry["slenderness"]),
        float(operational_summary["grasp_score_proxy"]),
        float(operational_summary["principal_curvature_score"]),
        float(operational_summary["size_score"]),
        float(physics["mass_proxy_kg"]) / 4.0,
        1.0 if physics["torque_sensitivity"] == "high" else 0.0,
    ]
    flag_values = [
        float(flags["finite_geometry"]),
        float(flags["finite_operational"]),
        float(flags["nondegenerate_extent"]),
        float(flags["pose_quaternion_present"]),
        float(flags["task_id_matches"]),
    ]
    return geometric + operational + scalar + flag_values + one_hot(entry["task"], TASKS) + one_hot(entry["dataset"], DATASETS)


def teacher_scores(entry: dict[str, Any]) -> dict[str, float]:
    category = entry["category"]
    task = entry["task"]
    geometry = entry["geometry_summary"]
    operational_summary = entry["operational_summary"]
    physics = entry["physics_prior"]
    flags = entry["quality_flags"]
    extents = np.asarray(geometry["extent_xyz"], dtype=np.float32)
    pose = np.asarray(operational_summary["target_pose_xyzwg"], dtype=np.float32)
    grasp = float(operational_summary["grasp_score_proxy"])
    aspect = float(geometry["aspect_ratio"])
    curvature = float(operational_summary["principal_curvature_score"])

    semantic = 0.90 if task in {"pick", "place"} else 0.74
    if task == "pour":
        semantic = 0.94 if category in CONTAINER_CATEGORIES else 0.54
    elif task == "tool_use":
        semantic = 0.88 if category in TOOL_LIKE_CATEGORIES or aspect > 2.0 else 0.58
    elif task == "insert":
        semantic = 0.86 if aspect < 3.0 else 0.66

    xy_radius = float(np.linalg.norm(pose[:2]))
    z = float(pose[2])
    max_extent = float(max(extents.max(), 1e-6))
    z_ok = 1.0 - min(abs(z - 0.35 * max(float(extents[2]), 1e-6)) / max_extent, 1.0)
    xy_ok = 1.0 - min(xy_radius / (0.45 * max_extent + 1e-6), 1.0)
    quat_norm = float(np.linalg.norm(pose[3:7]))
    quat_ok = 1.0 - min(abs(quat_norm - 1.0), 1.0)
    geometric_action = clamp01(0.30 * z_ok + 0.25 * xy_ok + 0.20 * quat_ok + 0.25 * clamp01(grasp * 2.5))

    flag_score = sum(float(value) for value in flags.values()) / max(len(flags), 1)
    mass = float(physics["mass_proxy_kg"])
    mass_ok = 1.0 if 0.2 <= mass <= 4.0 else 0.4
    contact_penalty = 0.08 if operational_summary["contact_risk"] == "high" and physics["torque_sensitivity"] != "high" else 0.0
    physics_feasibility = clamp01(0.55 * flag_score + 0.30 * mass_ok + 0.15 * (1.0 - contact_penalty))

    retrieval_utility = clamp01(0.35 * clamp01(grasp * 2.8) + 0.25 * curvature + 0.20 * min(aspect / 3.0, 1.0) + 0.20 * semantic)
    overall = clamp01(0.25 * semantic + 0.25 * geometric_action + 0.25 * physics_feasibility + 0.25 * retrieval_utility)
    return {
        "semantic_grounding": semantic,
        "geometric_action_consistency": geometric_action,
        "physics_feasibility": physics_feasibility,
        "retrieval_utility": retrieval_utility,
        "overall_quality": overall,
    }


class RAGQualityDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.entries = entries
        self.features = torch.tensor([feature_vector(entry) for entry in entries], dtype=torch.float32)
        self.targets = torch.tensor(
            [
                [
                    scores["semantic_grounding"],
                    scores["geometric_action_consistency"],
                    scores["physics_feasibility"],
                    scores["retrieval_utility"],
                    scores["overall_quality"],
                ]
                for scores in (teacher_scores(entry) for entry in entries)
            ],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"features": self.features[index], "targets": self.targets[index]}


def evaluate(model: RAGQualityAuditor, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    losses = []
    maes = []
    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            pred = model(features)
            losses.append(F.mse_loss(pred, targets).item())
            maes.append(torch.abs(pred - targets).mean().item())
    model.train()
    return {"loss": float(sum(losses) / max(len(losses), 1)), "mae": float(sum(maes) / max(len(maes), 1))}


def write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=True) + "\n")


def export_scored_entries(
    model: RAGQualityAuditor,
    dataset: RAGQualityDataset,
    out_dir: Path,
    device: torch.device,
    compression_ratio: float,
    diversity_lambda: float,
    uncertainty_lambda: float,
    mc_samples: int,
    min_task_fraction: float,
) -> dict[str, Any]:
    model.train()
    features = dataset.features.to(device)
    mc_preds = []
    with torch.no_grad():
        for _ in range(mc_samples):
            preds = []
            for start in range(0, len(dataset), 4096):
                preds.append(model(features[start : start + 4096]).cpu())
            mc_preds.append(torch.cat(preds, dim=0))
    stacked = torch.stack(mc_preds, dim=0)
    scores = stacked.mean(dim=0).numpy()
    uncertainty = stacked.std(dim=0).mean(dim=1).numpy()
    keys = [
        "semantic_grounding",
        "geometric_action_consistency",
        "physics_feasibility",
        "retrieval_utility",
        "overall_quality",
    ]
    scored = []
    for entry, row, unc in zip(dataset.entries, scores, uncertainty):
        quality = {key: float(value) for key, value in zip(keys, row)}
        quality["uncertainty"] = float(unc)
        quality["quality_confidence"] = clamp01(1.0 - float(unc))
        quality["compression_score"] = clamp01(float(quality["overall_quality"]) - uncertainty_lambda * float(unc))
        scored.append({**entry, "quality_audit": quality})
    keep = max(1, int(len(scored) * compression_ratio))
    compressed = select_diverse_entries(scored, keep, diversity_lambda, min_task_fraction)
    write_jsonl(out_dir / "scored_entries.jsonl", scored)
    write_jsonl(out_dir / "compressed_entries.jsonl", compressed)
    summary = {
        "num_scored_entries": len(scored),
        "num_compressed_entries": len(compressed),
        "compression_ratio": compression_ratio,
        "quality_mean_full": float(np.mean([entry["quality_audit"]["overall_quality"] for entry in scored])),
        "quality_mean_compressed": float(np.mean([entry["quality_audit"]["overall_quality"] for entry in compressed])),
        "uncertainty_mean_full": float(np.mean([entry["quality_audit"]["uncertainty"] for entry in scored])),
        "uncertainty_mean_compressed": float(np.mean([entry["quality_audit"]["uncertainty"] for entry in compressed])),
        "diversity_lambda": diversity_lambda,
        "uncertainty_lambda": uncertainty_lambda,
        "mc_samples": mc_samples,
        "min_task_fraction": min_task_fraction,
        "task_counts_compressed": dict(Counter(entry["task"] for entry in compressed)),
        "category_counts_compressed": dict(Counter(entry["category"] for entry in compressed).most_common(20)),
    }
    save_json(out_dir / "quality_manifest.json", summary)
    return summary


def select_diverse_entries(
    entries: list[dict[str, Any]],
    keep: int,
    diversity_lambda: float,
    min_task_fraction: float = 0.0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    dataset_counts: Counter[str] = Counter()
    remaining = sorted(entries, key=lambda item: item["quality_audit"]["compression_score"], reverse=True)
    floor = int(keep * min_task_fraction)
    if floor * len(TASKS) <= keep:
        for task in TASKS:
            task_pool = [entry for entry in remaining if entry["task"] == task]
            for entry in task_pool[:floor]:
                entry["quality_audit"]["selection_score"] = float(entry["quality_audit"]["compression_score"])
                selected.append(entry)
                task_counts[entry["task"]] += 1
                category_counts[entry["category"]] += 1
                dataset_counts[entry["dataset"]] += 1
        selected_ids = {entry["entry_id"] for entry in selected}
        remaining = [entry for entry in remaining if entry["entry_id"] not in selected_ids]
    target_task = keep / max(len(TASKS), 1)
    while remaining and len(selected) < keep:
        best_idx = 0
        best_score = -1.0
        for idx, entry in enumerate(remaining[: min(len(remaining), 2048)]):
            task_bonus = max(0.0, 1.0 - task_counts[entry["task"]] / max(target_task, 1.0))
            category_bonus = 1.0 / (1.0 + category_counts[entry["category"]])
            dataset_bonus = 1.0 / (1.0 + dataset_counts[entry["dataset"]] * 0.05)
            score = float(entry["quality_audit"]["compression_score"]) + diversity_lambda * (
                0.50 * task_bonus + 0.35 * category_bonus + 0.15 * dataset_bonus
            )
            if score > best_score:
                best_idx = idx
                best_score = score
        chosen = remaining.pop(best_idx)
        chosen["quality_audit"]["selection_score"] = float(best_score)
        selected.append(chosen)
        task_counts[chosen["task"]] += 1
        category_counts[chosen["category"]] += 1
        dataset_counts[chosen["dataset"]] += 1
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_quality_auditor.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    out_dir = ensure_dir(ROOT / cfg["artifacts_dir"] / cfg["run_name"])
    public_out = ensure_dir(ROOT / "public_data" / "s2p2e_rag_quality")

    entries = read_jsonl(ROOT / data_cfg["entries_path"])
    dataset = RAGQualityDataset(entries)
    generator = torch.Generator().manual_seed(cfg["seed"])
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_size = max(1, int(len(indices) * data_cfg.get("val_ratio", 0.1)))
    train_indices = indices[val_size:]
    val_indices = indices[:val_size]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=train_cfg["batch_size"], shuffle=False)
    device = torch.device(train_cfg["device"])
    model = RAGQualityAuditor(dataset.features.shape[1], train_cfg["hidden_dim"], depth=train_cfg.get("depth", 4)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    history = []
    best = {"epoch": 0, "val_loss": float("inf"), "val_mae": float("inf")}
    checkpoint_dir = ensure_dir(out_dir / "checkpoints")
    for epoch in range(train_cfg["epochs"]):
        logs = []
        model.train()
        for batch in train_loader:
            features = batch["features"].to(device)
            targets = batch["targets"].to(device)
            pred = model(features)
            loss = F.mse_loss(pred, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logs.append(float(loss.item()))
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_loss": float(sum(logs) / max(len(logs), 1)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row)
        if val["loss"] < best["val_loss"]:
            best = {"epoch": epoch + 1, "val_loss": val["loss"], "val_mae": val["mae"]}
            save_checkpoint(checkpoint_dir / "rag_quality_auditor_best.pt", {"model_state": model.state_dict(), "best": best})

    summary = export_scored_entries(
        model,
        dataset,
        public_out,
        device,
        data_cfg["compression_ratio"],
        data_cfg.get("diversity_lambda", 0.05),
        data_cfg.get("uncertainty_lambda", 0.25),
        data_cfg.get("mc_samples", 5),
        data_cfg.get("min_task_fraction", 0.0),
    )
    payload = {
        "entries": len(entries),
        "feature_dim": int(dataset.features.shape[1]),
        "train_samples": len(train_indices),
        "val_samples": len(val_indices),
        "best": best,
        "history": history,
        "compression": summary,
    }
    save_json(out_dir / "metrics.json", payload)

    source_csv = ROOT / "paper" / "source_data" / "rag_quality_compression_summary.csv"
    with source_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "value", "unit", "provenance"])
        writer.writerow(["full_entries", len(entries), "entries", "public_data/s2p2e_rag_dataset/entries.jsonl"])
        writer.writerow(["compressed_entries", summary["num_compressed_entries"], "entries", "public_data/s2p2e_rag_quality/compressed_entries.jsonl"])
        writer.writerow(["compression_ratio", summary["compression_ratio"], "fraction", "configs/rag_quality_auditor.yaml"])
        writer.writerow(["full_mean_quality", summary["quality_mean_full"], "score", "RAGQualityAuditor"])
        writer.writerow(["compressed_mean_quality", summary["quality_mean_compressed"], "score", "RAGQualityAuditor"])
        writer.writerow(["full_mean_uncertainty", summary["uncertainty_mean_full"], "score", "RAGQualityAuditor"])
        writer.writerow(["compressed_mean_uncertainty", summary["uncertainty_mean_compressed"], "score", "RAGQualityAuditor"])
        writer.writerow(["best_val_mae", best["val_mae"], "score", "artifacts/rag_quality_auditor/metrics.json"])

    print(payload)


if __name__ == "__main__":
    main()
