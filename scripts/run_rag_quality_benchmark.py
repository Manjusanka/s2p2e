from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.utils.config import load_config
from s2p2e.utils.io import ensure_dir, save_checkpoint, save_json
from s2p2e.utils.seed import set_seed

TASKS = ["pick", "place", "insert", "pour", "tool_use"]
DATASETS = ["objaverse", "shapenet"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def entry_index(entry: dict[str, Any]) -> int:
    return int(entry["entry_id"].rsplit("-", 1)[1])


def split_entries(entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    train, val, test = [], [], []
    for entry in entries:
        idx = entry_index(entry)
        if idx < 55168:
            train.append(entry)
        elif idx < 62064:
            val.append(entry)
        else:
            test.append(entry)
    return {"train": train, "val": val, "test": test}


def one_hot(value: str, choices: list[str]) -> list[float]:
    return [1.0 if value == choice else 0.0 for choice in choices]


def features(entry: dict[str, Any]) -> list[float]:
    geometry = entry["geometry_summary"]
    quality = entry["quality_audit"]
    return (
        [float(x) for x in entry["geometric"]]
        + one_hot(entry["task"], TASKS)
        + one_hot(entry["dataset"], DATASETS)
        + [
            float(geometry["volume_proxy"]),
            float(geometry["aspect_ratio"]),
            float(geometry["slenderness"]),
            float(quality["overall_quality"]),
            float(quality["uncertainty"]),
        ]
    )


def target(entry: dict[str, Any]) -> list[float]:
    return [float(x) for x in entry["operational_summary"]["target_pose_xyzwg"]]


class MemoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, entries: list[dict[str, Any]]) -> None:
        self.x = torch.tensor([features(entry) for entry in entries], dtype=torch.float32)
        self.y = torch.tensor([target(entry) for entry in entries], dtype=torch.float32)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[index], "y": self.y[index]}


class PoseMemoryPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        for _ in range(max(depth - 1, 0)):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU()]
        layers.append(nn.Linear(hidden_dim, 8))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raw = self.net(x)
        quat = F.normalize(raw[:, 3:7], dim=-1)
        return torch.cat([raw[:, :3], quat, torch.sigmoid(raw[:, 7:8])], dim=-1)


def summarize_subset(entries: list[dict[str, Any]], total: int) -> dict[str, float]:
    quality = [entry["quality_audit"]["overall_quality"] for entry in entries]
    uncertainty = [entry["quality_audit"]["uncertainty"] for entry in entries]
    return {
        "entries": float(len(entries)),
        "compression_ratio": float(len(entries) / max(total, 1)),
        "task_coverage": float(len(Counter(entry["task"] for entry in entries)) / len(TASKS)),
        "category_coverage": float(len(Counter(entry["category"] for entry in entries))),
        "mean_quality": float(np.mean(quality)),
        "mean_uncertainty": float(np.mean(uncertainty)),
    }


def select_qad(
    entries: list[dict[str, Any]],
    keep: int,
    diversity_lambda: float,
    min_task_fraction: float,
    pose_diversity_lambda: float = 0.0,
    pose_distribution_lambda: float = 0.0,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    task_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    pose_bin_counts: Counter[str] = Counter()
    full_pose_counts = Counter(pose_bin(entry) for entry in entries)
    target_pose_counts = {key: max(1.0, value / max(len(entries), 1) * keep) for key, value in full_pose_counts.items()}
    remaining = sorted(entries, key=base_selection_score, reverse=True)
    floor = int(keep * min_task_fraction)
    if floor * len(TASKS) <= keep:
        for task in TASKS:
            pool = [entry for entry in remaining if entry["task"] == task]
            for entry in pool[:floor]:
                selected.append(entry)
                task_counts[entry["task"]] += 1
                category_counts[entry["category"]] += 1
                pose_bin_counts[pose_bin(entry)] += 1
        selected_ids = {entry["entry_id"] for entry in selected}
        remaining = [entry for entry in remaining if entry["entry_id"] not in selected_ids]
    while remaining and len(selected) < keep:
        best_idx, best_score = 0, -1.0
        for idx, entry in enumerate(remaining[: min(len(remaining), 2048)]):
            task_bonus = 1.0 / (1.0 + task_counts[entry["task"]])
            category_bonus = 1.0 / (1.0 + category_counts[entry["category"]])
            pbin = pose_bin(entry)
            pose_bonus = 1.0 / (1.0 + pose_bin_counts[pbin])
            pose_target_bonus = max(0.0, 1.0 - pose_bin_counts[pbin] / target_pose_counts.get(pbin, 1.0))
            score = (
                base_selection_score(entry)
                + diversity_lambda * (0.6 * task_bonus + 0.4 * category_bonus)
                + pose_diversity_lambda * pose_bonus
                + pose_distribution_lambda * pose_target_bonus
            )
            if score > best_score:
                best_idx, best_score = idx, score
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        task_counts[chosen["task"]] += 1
        category_counts[chosen["category"]] += 1
        pose_bin_counts[pose_bin(chosen)] += 1
    return selected


def base_selection_score(entry: dict[str, Any]) -> float:
    quality = entry["quality_audit"]
    return float(quality.get("selection_score", quality.get("compression_score", quality["overall_quality"])))


def pose_bin(entry: dict[str, Any]) -> str:
    pose = entry["operational_summary"]["target_pose_xyzwg"]
    xy_radius = (float(pose[0]) ** 2 + float(pose[1]) ** 2) ** 0.5
    z = float(pose[2])
    gripper = float(pose[7])
    score = float(entry["quality_audit"]["retrieval_utility"])
    xy_bin = "near" if xy_radius < 0.08 else "midxy" if xy_radius < 0.18 else "far"
    z_bin = "low" if z < 0.25 else "mid" if z < 0.45 else "high"
    g_bin = "open" if gripper > 0.5 else "closed"
    u_bin = "utility_hi" if score >= 0.85 else "utility_lo"
    return f"{entry['task']}:{xy_bin}:{z_bin}:{g_bin}:{u_bin}"


def train_eval(
    train_entries: list[dict[str, Any]],
    test_entries: list[dict[str, Any]],
    cfg: dict[str, Any],
    out_dir: Path,
    label: str,
) -> dict[str, float]:
    train_ds = MemoryDataset(train_entries)
    test_ds = MemoryDataset(test_entries)
    device = torch.device(cfg["training"]["device"])
    model = PoseMemoryPredictor(train_ds.x.shape[1], cfg["model"]["hidden_dim"], cfg["model"]["depth"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["training"]["lr"], weight_decay=cfg["training"]["weight_decay"])
    loader = DataLoader(train_ds, batch_size=cfg["training"]["batch_size"], shuffle=True)
    for _ in range(cfg["training"]["epochs"]):
        model.train()
        for batch in loader:
            pred = model(batch["x"].to(device))
            y = batch["y"].to(device)
            loss = F.l1_loss(pred[:, :7], y[:, :7]) + 0.25 * F.l1_loss(pred[:, 7:8], y[:, 7:8])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    losses = []
    with torch.no_grad():
        for batch in DataLoader(test_ds, batch_size=cfg["training"]["batch_size"], shuffle=False):
            pred = model(batch["x"].to(device))
            y = batch["y"].to(device)
            losses.append(torch.abs(pred[:, :7] - y[:, :7]).mean().item())
    save_checkpoint(out_dir / f"{label}_pose_memory.pt", {"model_state": model.state_dict()})
    return {"pose_mae": float(np.mean(losses))}


def calibration_rows(entries: list[dict[str, Any]]) -> list[list[Any]]:
    rows = []
    bins = [(0.0, 0.75), (0.75, 0.82), (0.82, 0.88), (0.88, 0.92), (0.92, 1.01)]
    for low, high in bins:
        bucket = [entry for entry in entries if low <= entry["quality_audit"]["overall_quality"] < high]
        if not bucket:
            continue
        uncertainty = [entry["quality_audit"]["uncertainty"] for entry in bucket]
        reject = [1.0 if entry["quality_audit"]["overall_quality"] < 0.78 else 0.0 for entry in bucket]
        rows.append([f"{low:.2f}-{high:.2f}", len(bucket), float(np.mean(uncertainty)), float(np.mean(reject))])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/rag_quality_benchmark.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    rng = random.Random(cfg["seed"])
    out_dir = ensure_dir(ROOT / cfg["artifacts_dir"] / cfg["run_name"])
    source_dir = ROOT / "paper" / "source_data"
    entries = read_jsonl(ROOT / cfg["data"]["scored_entries"])
    splits = split_entries(entries)
    train_all = splits["train"]
    test_entries = splits["test"]
    total_train = len(train_all)
    rows = []
    for ratio in cfg["data"]["ratios"]:
        keep = max(1, int(total_train * ratio))
        methods = []
        if ratio == 1.0:
            methods.append(("full", train_all, 0))
        else:
            qad = select_qad(
                train_all,
                keep,
                cfg["data"]["diversity_lambda"],
                cfg["data"]["min_task_fraction"],
                cfg["data"].get("pose_diversity_lambda", 0.0),
                cfg["data"].get("pose_distribution_lambda", 0.0),
            )
            methods.append(("qa_d", qad, 0))
            for repeat in range(cfg["data"]["random_repeats"]):
                methods.append(("random", rng.sample(train_all, keep), repeat + 1))
        for method, subset, repeat in methods:
            label = f"{method}_{int(ratio * 100)}_r{repeat}"
            metrics = train_eval(subset, test_entries, cfg, out_dir, label)
            summary = summarize_subset(subset, total_train)
            rows.append(
                {
                    "method": method,
                    "ratio": ratio,
                    "repeat": repeat,
                    **summary,
                    **metrics,
                }
            )
            print(rows[-1])

    benchmark_csv = source_dir / "rag_quality_benchmark_curves.csv"
    with benchmark_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    calib_csv = source_dir / "rag_quality_calibration_bins.csv"
    with calib_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["quality_bin", "entries", "mean_uncertainty", "estimated_reject_rate"])
        writer.writerows(calibration_rows(entries))
    save_json(out_dir / "summary.json", {"rows": rows, "benchmark_csv": str(benchmark_csv), "calibration_csv": str(calib_csv)})
    print({"benchmark_csv": str(benchmark_csv), "calibration_csv": str(calib_csv), "runs": len(rows)})


if __name__ == "__main__":
    main()
