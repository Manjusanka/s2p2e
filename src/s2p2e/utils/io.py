from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=True)


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    torch.save(payload, Path(path))
