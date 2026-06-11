from pathlib import Path
import sys
import argparse

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.public_datasets import PublicRobotTransitionDataset, build_synthetic_public_dataset, load_droid_jsonl
from s2p2e.models.actor_critic import PhysicsInjectedActorCritic
from s2p2e.trainers.actor_critic import train_actor_critic_epoch
from s2p2e.utils.config import load_config
from s2p2e.utils.io import ensure_dir, save_checkpoint, save_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/public_robot_data.yaml")
    parser.add_argument("--out-dir", default="artifacts/actor_critic_droid")
    args = parser.parse_args()

    cfg = load_config(ROOT / args.config)
    train_cfg = cfg["training"]
    droid_root = ROOT / "public_data" / "droid_subset_jsonl"
    if droid_root.exists():
        samples = load_droid_jsonl(droid_root)
    else:
        samples = build_synthetic_public_dataset(train_cfg.get("synthetic_samples", 2048))
    out_dir = ensure_dir(ROOT / args.out_dir)
    dataset = PublicRobotTransitionDataset(samples)
    loader = DataLoader(dataset, batch_size=train_cfg["actor_batch_size"], shuffle=True)
    sample = samples[0]
    obs_dim = len(sample["obs"])
    action_dim = len(sample["action"])
    physics_dim = len(sample["physics_feat"])
    model = PhysicsInjectedActorCritic(obs_dim=obs_dim, action_dim=action_dim, physics_dim=physics_dim)
    device = torch.device(train_cfg.get("device", "cpu"))
    model.to(device)
    opt_actor = torch.optim.AdamW(model.actor.parameters(), lr=train_cfg["actor_lr"])
    opt_critic = torch.optim.AdamW(model.critic.parameters(), lr=train_cfg.get("critic_lr", train_cfg["actor_lr"]))
    history = []
    for epoch in range(train_cfg["actor_epochs"]):
        metrics = train_actor_critic_epoch(
            model,
            loader,
            opt_actor,
            opt_critic,
            device,
            {
                "gamma": train_cfg["gamma"],
                "tau": train_cfg["tau"],
                "torque_limit": train_cfg.get("torque_limit", 12.0),
                "lambda_viol": train_cfg.get("lambda_viol", 0.5),
            },
        )
        history.append({"epoch": epoch + 1, **metrics})
    save_checkpoint(out_dir / "actor_critic_droid_best.pt", {"model_state": model.state_dict(), "history": history})
    save_json(out_dir / "actor_critic_droid_history.json", {"history": history, "num_samples": len(samples)})
    print({"history": history, "num_samples": len(samples)})


if __name__ == "__main__":
    main()
