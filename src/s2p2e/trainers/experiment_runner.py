from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from s2p2e.trainers.pipeline import run_pipeline
from s2p2e.utils.config import load_config
from s2p2e.utils.io import save_json


def run_single_experiment(config_path: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    summary = run_pipeline(config)
    return {"config": str(config_path), "run_name": config["run_name"], "summary": summary}


def run_experiment_suite(suite_path: str | Path) -> dict[str, Any]:
    with Path(suite_path).open("r", encoding="utf-8") as handle:
        suite = yaml.safe_load(handle)
    results = []
    for config_path in suite["experiments"]:
        results.append(run_single_experiment(config_path))
    payload = {"suite": str(suite_path), "results": results}
    out_path = Path("artifacts") / "experiment_suite_results.json"
    save_json(out_path, payload)
    return payload
