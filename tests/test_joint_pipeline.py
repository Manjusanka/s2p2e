from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.cache import PointCloudCache
from s2p2e.data.corpus import build_object_corpus
from s2p2e.data.datasets import JointStackDataset, build_tokenizer, collate_joint
from s2p2e.models.full_stack import FullS2P2E
from s2p2e.models.layer1 import Layer1Model
from s2p2e.models.layer2 import PhysicsPriorEncoder
from s2p2e.models.layer3 import ResidualController
from s2p2e.models.router import TaskWeightRouter
from s2p2e.trainers.joint import eval_joint_epoch


def test_joint_forward_and_eval() -> None:
    records = build_object_corpus(ROOT / "shapenet" / "models", ROOT / "objaverse" / "models", 1, 0)
    tasks = ["pick", "place"]
    tokenizer = build_tokenizer(records, tasks)
    cache = PointCloudCache(ROOT / "artifacts" / "test_cache", 64)
    dataset = JointStackDataset(records[:2], tasks, tokenizer, 64, 8, 7, cache=cache)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_joint)

    layer1 = Layer1Model(tokenizer.vocab_size, embedding_dim=128, retrieval_topk=2)
    layer2 = PhysicsPriorEncoder(state_dim=21, cond_dim=21, dof=7, hidden_dim=64, layers=4)
    layer3 = ResidualController(dof=7, history_length=8)
    router = TaskWeightRouter({"pick": [0.3, 0.5, 0.2], "place": [0.3, 0.4, 0.3]}, tasks)
    model = FullS2P2E(layer1, layer2, layer3, router, dof=7)

    metrics = eval_joint_epoch(
        model,
        loader,
        torch.device("cpu"),
        {"lambda_joint_pose": 3.0, "lambda_joint_tau": 1.0, "lambda_joint_feasibility": 0.5},
    )
    assert "loss" in metrics
