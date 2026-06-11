from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.corpus import build_object_corpus
from s2p2e.data.datasets import Layer1Dataset, build_tokenizer, collate_layer1
from s2p2e.models.layer1 import Layer1Model
from s2p2e.models.layer2 import PhysicsPriorEncoder
from s2p2e.models.layer3 import ResidualController


def test_corpus_builds() -> None:
    records = build_object_corpus(ROOT / "shapenet" / "models", ROOT / "objaverse" / "models", 1, 1)
    assert len(records) >= 2


def test_layer1_forward_shapes() -> None:
    records = build_object_corpus(ROOT / "shapenet" / "models", ROOT / "objaverse" / "models", 1, 0)
    tasks = ["pick", "place"]
    tokenizer = build_tokenizer(records, tasks)
    dataset = Layer1Dataset(records[:2], tasks, tokenizer, num_points=64, split="train")
    batch = collate_layer1([dataset[0], dataset[1]])
    model = Layer1Model(tokenizer.vocab_size, embedding_dim=128, retrieval_topk=3)
    outputs = model(batch["points"], batch["tokens"], batch["geometric"], batch["operational"])
    assert outputs["poses"].shape == (2, 3, 8)
    assert outputs["retrieval_logits"].shape == (2, 2)


def test_layer2_and_layer3_shapes() -> None:
    ppe = PhysicsPriorEncoder(state_dim=21, cond_dim=21, dof=7, hidden_dim=64, layers=4)
    state = torch.randn(10, 21)
    cond = torch.randn(10, 21)
    outputs = ppe(state, cond)
    assert outputs["tau"].shape == (10, 7)

    controller = ResidualController(dof=7, history_length=8, clip_ratio=0.2)
    ctrl_out = controller(
        torch.randn(4, 8, 7),
        torch.randn(4, 8, 7),
        torch.randn(4, 8, 7),
        torch.randn(4, 6),
        torch.randn(4, 7),
        torch.randn(4, 5),
        torch.randn(4, 7).abs() + 0.1,
    )
    assert ctrl_out["command"].shape == (4, 7)
