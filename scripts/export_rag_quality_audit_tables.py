from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def sd(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    return (sum((value - mu) ** 2 for value in values) / (len(values) - 1)) ** 0.5


def bucket(value: float) -> str:
    if value >= 0.90:
        return "accept"
    if value >= 0.78:
        return "revise"
    return "reject"


def write_audit_sample(full: list[dict], compressed: list[dict], out_path: Path, n: int = 500) -> None:
    rng = random.Random(42)
    compressed_ids = {entry["entry_id"] for entry in compressed}
    full_pool = rng.sample(full, min(n // 2, len(full)))
    compressed_pool = rng.sample(compressed, min(n - len(full_pool), len(compressed)))
    rows = []
    for source_split, entries in [("full", full_pool), ("compressed", compressed_pool)]:
        for entry in entries:
            quality = entry["quality_audit"]
            rows.append(
                {
                    "audit_id": f"audit_{len(rows):04d}",
                    "entry_id": entry["entry_id"],
                    "source_split": source_split,
                    "also_in_compressed": entry["entry_id"] in compressed_ids,
                    "dataset": entry["dataset"],
                    "category": entry["category"],
                    "task": entry["task"],
                    "instruction": entry["instruction"]["canonical"],
                    "semantic_grounding_score": quality["semantic_grounding"],
                    "geometric_action_consistency_score": quality["geometric_action_consistency"],
                    "physics_feasibility_score": quality["physics_feasibility"],
                    "retrieval_utility_score": quality["retrieval_utility"],
                    "overall_quality_score": quality["overall_quality"],
                    "uncertainty": quality["uncertainty"],
                    "predicted_decision": bucket(quality["overall_quality"]),
                    "human_semantic_grounding_score": "",
                    "human_geometric_action_consistency_score": "",
                    "human_physics_feasibility_score": "",
                    "human_retrieval_utility_score": "",
                    "human_overall_decision": "",
                    "adjudication_notes": "",
                    "provenance": "provisional_model_estimate",
                }
            )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_summary(full: list[dict], compressed: list[dict], out_path: Path) -> None:
    rows = []
    for name, entries in [("full_scored", full), ("qa_d_compressed", compressed)]:
        quality = [entry["quality_audit"]["overall_quality"] for entry in entries]
        uncertainty = [entry["quality_audit"]["uncertainty"] for entry in entries]
        accept = [1.0 if bucket(entry["quality_audit"]["overall_quality"]) == "accept" else 0.0 for entry in entries]
        revise = [1.0 if bucket(entry["quality_audit"]["overall_quality"]) == "revise" else 0.0 for entry in entries]
        reject = [1.0 if bucket(entry["quality_audit"]["overall_quality"]) == "reject" else 0.0 for entry in entries]
        rows.extend(
            [
                [name, "entries", len(entries), "count", "provisional_model_estimate"],
                [name, "mean_overall_quality", mean(quality), "score", "provisional_model_estimate"],
                [name, "sd_overall_quality", sd(quality), "score", "provisional_model_estimate"],
                [name, "mean_uncertainty", mean(uncertainty), "score", "provisional_model_estimate"],
                [name, "predicted_accept_rate", mean(accept), "fraction", "provisional_model_estimate"],
                [name, "predicted_revise_rate", mean(revise), "fraction", "provisional_model_estimate"],
                [name, "predicted_reject_rate", mean(reject), "fraction", "provisional_model_estimate"],
            ]
        )
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "metric", "value", "unit", "provenance"])
        writer.writerows(rows)


def main() -> None:
    quality_dir = ROOT / "public_data" / "s2p2e_rag_quality"
    source_dir = ROOT / "paper" / "source_data"
    full = read_jsonl(quality_dir / "scored_entries.jsonl")
    compressed = read_jsonl(quality_dir / "compressed_entries.jsonl")
    write_audit_sample(full, compressed, source_dir / "rag_quality_human_audit_template.csv")
    write_summary(full, compressed, source_dir / "rag_quality_predicted_audit_summary.csv")
    print(
        {
            "audit_template": str(source_dir / "rag_quality_human_audit_template.csv"),
            "summary": str(source_dir / "rag_quality_predicted_audit_summary.csv"),
            "sample_n": 500,
        }
    )


if __name__ == "__main__":
    main()
