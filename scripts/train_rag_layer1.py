from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.cache import PointCloudCache
from s2p2e.data.corpus import build_object_corpus
from s2p2e.data.datasets import Layer1Dataset, build_tokenizer, collate_layer1, split_records
from s2p2e.data.kb import KnowledgeBaseBuilder
from s2p2e.models.layer1 import Layer1Model
from s2p2e.trainers.pipeline import eval_layer1, run_with_best, train_layer1
from s2p2e.utils.config import load_config
from s2p2e.utils.io import ensure_dir, save_json
from s2p2e.utils.seed import set_seed


def _resolve_roots(value: str | list[str]) -> Path | list[Path]:
    if isinstance(value, list):
        return [ROOT / path for path in value]
    return ROOT / value


def _load_compatible(model: torch.nn.Module, checkpoint_path: Path) -> dict[str, int]:
    if not checkpoint_path.exists():
        return {"loaded": 0, "skipped": 0}
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    source = checkpoint.get("model_state", checkpoint)
    target = model.state_dict()
    compatible = {key: value for key, value in source.items() if key in target and target[key].shape == value.shape}
    skipped = len(source) - len(compatible)
    target.update(compatible)
    model.load_state_dict(target)
    return {"loaded": len(compatible), "skipped": skipped}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_rag_full.yaml")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    set_seed(cfg["seed"])
    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]

    records = build_object_corpus(
        _resolve_roots(data_cfg.get("shapenet_roots", data_cfg.get("shapenet_root"))),
        ROOT / data_cfg["objaverse_root"],
        data_cfg["shapenet_limit_per_category"],
        data_cfg["objaverse_limit"],
    )
    splits = split_records(records, data_cfg["train_ratio"], data_cfg["val_ratio"])
    tokenizer = build_tokenizer(records, data_cfg["tasks"])
    cache = PointCloudCache(ROOT / data_cfg["cache_root"], data_cfg["num_points"])

    run_dir = ensure_dir(ROOT / cfg["artifacts_dir"] / cfg["run_name"])
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    kb_dir = ensure_dir(run_dir / "kb")
    metric_dir = ensure_dir(run_dir / "metrics")

    kb_entries = KnowledgeBaseBuilder(records, data_cfg["tasks"], data_cfg["num_points"]).build()
    KnowledgeBaseBuilder(records, data_cfg["tasks"], data_cfg["num_points"]).export_json(kb_dir, kb_entries)

    train_dataset = Layer1Dataset(splits["train"], data_cfg["tasks"], tokenizer, data_cfg["num_points"], "train", cache=cache)
    val_dataset = Layer1Dataset(splits["val"] or splits["train"][:1], data_cfg["tasks"], tokenizer, data_cfg["num_points"], "val", cache=cache)
    loader_kwargs = {
        "batch_size": train_cfg["batch_size"],
        "num_workers": train_cfg["num_workers"],
    }
    train_loader = DataLoader(train_dataset, shuffle=True, collate_fn=collate_layer1, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, collate_fn=collate_layer1, **loader_kwargs)

    device = torch.device(train_cfg["device"])
    model = Layer1Model(tokenizer.vocab_size, model_cfg["embedding_dim"], model_cfg["retrieval_topk"]).to(device)
    resume_stats = _load_compatible(model, ROOT / train_cfg["resume_checkpoint"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr_layer1"], weight_decay=train_cfg["weight_decay"])
    best = run_with_best(
        "layer1",
        train_cfg["epochs_layer1"],
        train_layer1,
        eval_layer1,
        model,
        train_loader,
        val_loader,
        optimizer,
        device,
        train_cfg,
        checkpoint_dir,
    )
    summary = {
        "records": len(records),
        "kb_entries": len(kb_entries),
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "vocab_size": tokenizer.vocab_size,
        "resume": resume_stats,
        "best": best,
    }
    save_json(metric_dir / "rag_layer1_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
