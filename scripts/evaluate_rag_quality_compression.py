from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def summarize(entries: list[dict]) -> dict[str, float]:
    return {
        "entries": float(len(entries)),
        "tasks": float(len(Counter(entry["task"] for entry in entries))),
        "categories": float(len(Counter(entry["category"] for entry in entries))),
        "mean_quality": float(sum(entry["quality_audit"]["overall_quality"] for entry in entries) / max(len(entries), 1)),
        "mean_uncertainty": float(sum(entry["quality_audit"]["uncertainty"] for entry in entries) / max(len(entries), 1)),
    }


def main() -> None:
    quality_dir = ROOT / "public_data" / "s2p2e_rag_quality"
    full = read_jsonl(quality_dir / "scored_entries.jsonl")
    compressed = read_jsonl(quality_dir / "compressed_entries.jsonl")
    full_summary = summarize(full)
    compressed_summary = summarize(compressed)
    out_path = ROOT / "paper" / "source_data" / "rag_quality_compression_eval.csv"
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["split", "entries", "tasks", "categories", "mean_quality", "mean_uncertainty"])
        writer.writerow(["full", full_summary["entries"], full_summary["tasks"], full_summary["categories"], full_summary["mean_quality"], full_summary["mean_uncertainty"]])
        writer.writerow(["compressed", compressed_summary["entries"], compressed_summary["tasks"], compressed_summary["categories"], compressed_summary["mean_quality"], compressed_summary["mean_uncertainty"]])
    print({"output": str(out_path), "full": full_summary, "compressed": compressed_summary})


if __name__ == "__main__":
    main()
