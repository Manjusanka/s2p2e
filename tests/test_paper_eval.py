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
from s2p2e.data.datasets import JointStackDataset, Layer1Dataset, build_tokenizer, collate_joint, collate_layer1
from s2p2e.models.full_stack import FullS2P2E
from s2p2e.models.layer1 import Layer1Model
from s2p2e.models.layer2 import PhysicsPriorEncoder
from s2p2e.models.layer3 import ResidualController
from s2p2e.models.router import TaskWeightRouter
from s2p2e.trainers.paper_eval import evaluate_kb_tiers, evaluate_task_families


def test_paper_eval_helpers() -> None:
    records = build_object_corpus(ROOT / "shapenet" / "models", ROOT / "objaverse" / "models", 1, 0)
    tasks = ["pick", "place"]
    tokenizer = build_tokenizer(records, tasks)
    cache = PointCloudCache(ROOT / "artifacts" / "test_cache_eval", 64)

    layer1_ds = Layer1Dataset(records[:2], tasks, tokenizer, 64, "val", cache=cache)
    layer1_loader = DataLoader(layer1_ds, batch_size=2, collate_fn=collate_layer1)
    layer1 = Layer1Model(tokenizer.vocab_size, embedding_dim=128, retrieval_topk=2)
    kb_results = evaluate_kb_tiers(layer1, layer1_loader, torch.device("cpu"), {"lambda_pose": 5.0, "lambda_gripper": 1.0, "lambda_retrieval": 0.2})
    assert "full_3tier" in kb_results

    joint_ds = JointStackDataset(records[:2], tasks, tokenizer, 64, 8, 7, cache=cache)
    joint_loader = DataLoader(joint_ds, batch_size=2, collate_fn=collate_joint)
    layer2 = PhysicsPriorEncoder(state_dim=21, cond_dim=21, dof=7, hidden_dim=64, layers=4)
    layer3 = ResidualController(dof=7, history_length=8)
    router = TaskWeightRouter({"pick": [0.3, 0.5, 0.2], "place": [0.3, 0.4, 0.3]}, tasks)
    full = FullS2P2E(layer1, layer2, layer3, router, dof=7)
    task_results = evaluate_task_families(full, joint_loader, torch.device("cpu"), {}, tasks)
    assert "pick" in task_results or "place" in task_results
