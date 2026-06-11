from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader

from s2p2e.data.public_datasets import PublicRobotTransitionDataset, load_droid_jsonl
from s2p2e.models.actor_critic import PhysicsInjectedActorCritic
from s2p2e.trainers.actor_critic import train_actor_critic_epoch
from s2p2e.trainers.pipeline import run_pipeline
from s2p2e.utils.config import load_config
from s2p2e.utils.io import ensure_dir, save_checkpoint, save_json


def run_full_pipeline(config_path: str | Path) -> dict:
    cfg = load_config(config_path)
    root = Path.cwd()
    artifacts_root = ensure_dir(root / cfg["artifacts_dir"] / cfg["run_name"])
    stage_results = {}

    s2p2e_summary = {}
    if cfg["stages"].get("train_s2p2e", True):
        s2p2e_cfg = load_config(root / cfg["s2p2e_config"])
        s2p2e_summary = run_pipeline(s2p2e_cfg)
        stage_results["s2p2e"] = s2p2e_summary

    droid_root = root / "public_data" / "droid_subset_jsonl"
    if cfg["stages"].get("train_actor_critic_droid", True) and droid_root.exists():
        samples = load_droid_jsonl(droid_root)
        dataset = PublicRobotTransitionDataset(samples)
        loader = DataLoader(dataset, batch_size=cfg["policy"]["batch_size"], shuffle=True)
        sample = samples[0]
        model = PhysicsInjectedActorCritic(
            obs_dim=len(sample["obs"]),
            action_dim=len(sample["action"]),
            physics_dim=len(sample["physics_feat"]),
        )
        device = torch.device("cpu")
        model.to(device)
        opt_actor = torch.optim.AdamW(model.actor.parameters(), lr=cfg["policy"]["actor_lr"])
        opt_critic = torch.optim.AdamW(model.critic.parameters(), lr=cfg["policy"]["critic_lr"])
        history = []
        for epoch in range(cfg["policy"]["epochs"]):
            metrics = train_actor_critic_epoch(
                model,
                loader,
                opt_actor,
                opt_critic,
                device,
                {
                    "gamma": cfg["policy"]["gamma"],
                    "tau": cfg["policy"]["tau"],
                    "torque_limit": cfg["policy"]["torque_limit"],
                    "lambda_viol": cfg["policy"]["lambda_viol"],
                },
            )
            history.append({"epoch": epoch + 1, **metrics})
        save_checkpoint(artifacts_root / "policy_bridge_best.pt", {"model_state": model.state_dict(), "history": history})
        save_json(artifacts_root / "policy_bridge_history.json", {"history": history, "num_samples": len(samples)})
        stage_results["policy_bridge"] = {"history": history, "num_samples": len(samples)}

    save_json(artifacts_root / "full_pipeline_summary.json", stage_results)
    return stage_results
