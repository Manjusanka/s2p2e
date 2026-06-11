from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from s2p2e.data.corpus import ObjectRecord
from s2p2e.data.synthetic import build_geometry_targets, build_prompt
from s2p2e.utils.geometry import (
    load_objaverse_glb,
    load_shapenet_txt,
    normalize_point_cloud,
    sample_or_pad,
)
from s2p2e.utils.io import ensure_dir, save_json
from s2p2e.utils.text import SimpleTokenizer


@dataclass
class KBEntry:
    object_id: str
    dataset: str
    category: str
    task: str
    semantic_text: str
    geometric: list[float]
    operational: list[float]


class KnowledgeBaseBuilder:
    def __init__(self, records: list[ObjectRecord], tasks: list[str], num_points: int) -> None:
        self.records = records
        self.tasks = tasks
        self.num_points = num_points

    def _load_points(self, record: ObjectRecord, seed: int) -> np.ndarray:
        rng = np.random.default_rng(seed)
        if record.dataset == "shapenet":
            points = load_shapenet_txt(record.path)
            points = sample_or_pad(points, self.num_points, rng)
        else:
            points = load_objaverse_glb(record.path, self.num_points, rng)
        return normalize_point_cloud(points).astype(np.float32)

    def build(self) -> list[KBEntry]:
        entries: list[KBEntry] = []
        for idx, record in enumerate(self.records):
            try:
                points = self._load_points(record, idx + 11)
            except Exception:
                continue
            for task_id, task in enumerate(self.tasks):
                targets = build_geometry_targets(points, task_id)
                entries.append(
                    KBEntry(
                        object_id=record.object_id,
                        dataset=record.dataset,
                        category=record.category,
                        task=task,
                        semantic_text=f"{build_prompt(record.category, task)} {record.affordance_text}",
                        geometric=targets.geometric.tolist(),
                        operational=targets.operational.tolist(),
                    )
                )
        return entries

    def build_tokenizer(self, entries: list[KBEntry]) -> SimpleTokenizer:
        return SimpleTokenizer([entry.semantic_text for entry in entries])

    def export_json(self, out_dir: str | Path, entries: list[KBEntry]) -> None:
        out_dir = ensure_dir(out_dir)
        save_json(out_dir / "kb_entries.json", {"entries": [asdict(entry) for entry in entries]})


def encode_kb_entries(
    entries: list[KBEntry],
    tokenizer: SimpleTokenizer,
    text_encoder: torch.nn.Module,
    geom_encoder: torch.nn.Module,
    op_encoder: torch.nn.Module,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    text_encoder.eval()
    geom_encoder.eval()
    op_encoder.eval()
    semantic_tokens = torch.stack([tokenizer.encode(entry.semantic_text) for entry in entries], dim=0).to(device)
    geometric = torch.tensor([entry.geometric for entry in entries], dtype=torch.float32, device=device)
    operational = torch.tensor([entry.operational for entry in entries], dtype=torch.float32, device=device)
    with torch.no_grad():
        semantic = text_encoder(semantic_tokens)
        geom = geom_encoder(geometric)
        ops = op_encoder(operational)
        bank = (semantic + geom + ops) / 3.0
    return {"semantic": semantic.cpu(), "geometric": geom.cpu(), "operational": ops.cpu(), "bank": bank.cpu()}
