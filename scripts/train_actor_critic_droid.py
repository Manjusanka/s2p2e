from pathlib import Path
import sys
import argparse

import torch
from torch.utils.data import DataLoader, Subset

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.public_datasets import (
    PublicRobotTransitionDataset,
    build_cross_embodiment_public_dataset,
    build_synthetic_public_dataset,
    load_droid_jsonl,
)
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
    source_counts = {}
    if droid_root.exists():
        samples = load_droid_jsonl(droid_root)
        for sample in samples:
            sample.setdefault("source", "droid")
        source_counts["droid"] = len(samples)
    else:
        samples = build_synthetic_public_dataset(train_cfg.get("synthetic_samples", 2048))
        for sample in samples:
            sample.setdefault("source", "synthetic_fallback")
        source_counts["synthetic_fallback"] = len(samples)
    if train_cfg.get("cross_embodiment_samples", 0) > 0:
        cross_samples = build_cross_embodiment_public_dataset(
            train_cfg["cross_embodiment_samples"],
            seed=train_cfg.get("seed", 42),
        )
        samples.extend(cross_samples)
        source_counts["synthetic_cross_embodiment"] = len(cross_samples)
    out_dir = ensure_dir(ROOT / args.out_dir)
    dataset = PublicRobotTransitionDataset(samples)
    generator = torch.Generator().manual_seed(train_cfg.get("seed", 42))
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_size = max(1, int(len(indices) * train_cfg.get("val_ratio", 0.1)))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=train_cfg["actor_batch_size"],
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=train_cfg["actor_batch_size"], shuffle=False)
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
    best = {"epoch": 0, "critic_loss": float("inf")}
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
        val_metrics = evaluate_actor_critic(model, val_loader, device, train_cfg)
        row = {"epoch": epoch + 1, **metrics, **{f"val_{key}": value for key, value in val_metrics.items()}}
        history.append(row)
        if val_metrics["critic_loss"] < best["critic_loss"]:
            best = {"epoch": epoch + 1, "critic_loss": val_metrics["critic_loss"]}
            save_checkpoint(out_dir / "actor_critic_droid_best.pt", {"model_state": model.state_dict(), "history": history, "best": best})
    save_json(
        out_dir / "actor_critic_droid_history.json",
        {
            "history": history,
            "best": best,
            "num_samples": len(samples),
            "train_samples": len(train_indices),
            "val_samples": len(val_indices),
            "source_counts": source_counts,
        },
    )
    print({"history": history, "best": best, "num_samples": len(samples), "source_counts": source_counts})


@torch.no_grad()
def evaluate_actor_critic(model, loader, device, train_cfg):
    model.eval()
    losses = []
    gamma = train_cfg.get("gamma", 0.99)
    torque_limit = train_cfg.get("torque_limit", 12.0)
    lambda_viol = train_cfg.get("lambda_viol", 0.5)
    for batch in loader:
        batch = {key: value.to(device) for key, value in batch.items()}
        shaped_reward = model.shape_reward(batch["reward"], batch["predicted_tau"], torque_limit, lambda_viol)
        next_action = model.act(batch["next_obs"], batch["physics_feat"])
        target_q = shaped_reward + gamma * model.target_critic(batch["next_obs"], next_action, batch["physics_feat"])
        current_q = model.evaluate(batch["obs"], batch["action"], batch["physics_feat"])
        losses.append(torch.nn.functional.mse_loss(current_q, target_q).item())
    model.train()
    return {"critic_loss": float(sum(losses) / max(len(losses), 1))}


if __name__ == "__main__":
    main()
