import json
from pathlib import Path

import torch

from scripts.train_rag_quality_auditor import RAGQualityDataset, select_diverse_entries, teacher_scores
from s2p2e.models.rag_quality import RAGQualityAuditor


ROOT = Path(__file__).resolve().parents[1]


def test_rag_quality_teacher_scores_and_model_shapes():
    entry = json.loads((ROOT / "public_data" / "s2p2e_rag_dataset" / "entries.jsonl").read_text(encoding="utf-8").splitlines()[0])
    scores = teacher_scores(entry)
    dataset = RAGQualityDataset([entry])
    model = RAGQualityAuditor(dataset.features.shape[1], hidden_dim=32)

    assert set(scores) == {
        "semantic_grounding",
        "geometric_action_consistency",
        "physics_feasibility",
        "retrieval_utility",
        "overall_quality",
    }
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert model(dataset.features).shape == (1, 5)


def test_diverse_selection_keeps_multiple_tasks():
    entry = json.loads((ROOT / "public_data" / "s2p2e_rag_dataset" / "entries.jsonl").read_text(encoding="utf-8").splitlines()[0])
    entries = []
    for idx, task in enumerate(["pick", "place", "insert", "pour", "tool_use"] * 2):
        item = {**entry, "task": task, "quality_audit": {"compression_score": 0.8 - idx * 0.01}}
        entries.append(item)

    selected = select_diverse_entries(entries, keep=5, diversity_lambda=0.1, min_task_fraction=0.2)

    assert len(selected) == 5
    assert len({item["task"] for item in selected}) == 5
    assert all("selection_score" in item["quality_audit"] for item in selected)
