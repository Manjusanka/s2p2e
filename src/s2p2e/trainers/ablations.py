from __future__ import annotations

from pathlib import Path

from s2p2e.utils.io import save_json


def export_ablation_results(path: str | Path, results: dict) -> None:
    save_json(path, {"ablations": results})
