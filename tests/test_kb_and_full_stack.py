from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.corpus import build_object_corpus
from s2p2e.data.datasets import Layer1Dataset, build_tokenizer, collate_layer1
from s2p2e.data.kb import KnowledgeBaseBuilder
from s2p2e.models.full_stack import FullS2P2E
from s2p2e.models.layer1 import Layer1Model
from s2p2e.models.layer2 import PhysicsPriorEncoder
from s2p2e.models.layer3 import ResidualController
from s2p2e.models.router import TaskWeightRouter


def test_kb_builder_exports_entries() -> None:
    records = build_object_corpus(ROOT / "shapenet" / "models", ROOT / "objaverse" / "models", 1, 0)
    builder = KnowledgeBaseBuilder(records[:2], ["pick", "place"], 64)
    entries = builder.build()
    assert len(entries) == 4


def test_full_stack_forward() -> None:
    records = build_object_corpus(ROOT / "shapenet" / "models", ROOT / "objaverse" / "models", 1, 0)
    tasks = ["pick", "place"]
    tokenizer = build_tokenizer(records, tasks)
    dataset = Layer1Dataset(records[:2], tasks, tokenizer, num_points=64, split="train")
    batch = collate_layer1([dataset[0], dataset[1]])

    layer1 = Layer1Model(tokenizer.vocab_size, embedding_dim=128, retrieval_topk=2)
    layer2 = PhysicsPriorEncoder(state_dim=21, cond_dim=21, dof=7, hidden_dim=64, layers=4)
    layer3 = ResidualController(dof=7, history_length=8)
    router = TaskWeightRouter({"pick": [0.3, 0.5, 0.2], "place": [0.3, 0.4, 0.3]}, tasks)
    model = FullS2P2E(layer1, layer2, layer3, router)

    outputs = model(
        batch["points"],
        batch["tokens"],
        batch["task_id"],
        batch["geometric"],
        batch["operational"],
        torch.randn(2, 21),
        torch.randn(2, 8, 7),
        torch.randn(2, 8, 7),
        torch.randn(2, 8, 7),
        torch.randn(2, 6),
        torch.randn(2, 7),
        torch.randn(2, 5),
        torch.randn(2, 7).abs() + 0.1,
    )
    assert outputs["q_ref"].shape == (2, 7)
    assert outputs["controller"]["command"].shape == (2, 7)
