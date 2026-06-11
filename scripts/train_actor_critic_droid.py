from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from s2p2e.data.public_datasets import PublicRobotTransitionDataset, build_synthetic_public_dataset, load_droid_jsonl
from s2p2e.models.actor_critic import PhysicsInjectedActorCritic
from s2p2e.trainers.actor_critic import train_actor_critic_epoch
from s2p2e.utils.io import ensure_dir, save_checkpoint, save_json


def main() -> None:
    droid_root = ROOT / "public_data" / "droid_subset_jsonl"
    if droid_root.exists():
        samples = load_droid_jsonl(droid_root)
    else:
        samples = build_synthetic_public_dataset()
    out_dir = ensure_dir(ROOT / "artifacts" / "actor_critic_droid")
    dataset = PublicRobotTransitionDataset(samples)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    sample = samples[0]
    obs_dim = len(sample["obs"])
    action_dim = len(sample["action"])
    physics_dim = len(sample["physics_feat"])
    model = PhysicsInjectedActorCritic(obs_dim=obs_dim, action_dim=action_dim, physics_dim=physics_dim)
    device = torch.device("cpu")
    model.to(device)
    opt_actor = torch.optim.AdamW(model.actor.parameters(), lr=3e-4)
    opt_critic = torch.optim.AdamW(model.critic.parameters(), lr=3e-4)
    history = []
    for epoch in range(2):
        metrics = train_actor_critic_epoch(
            model,
            loader,
            opt_actor,
            opt_critic,
            device,
            {"gamma": 0.99, "tau": 0.005, "torque_limit": 12.0, "lambda_viol": 0.5},
        )
        history.append({"epoch": epoch + 1, **metrics})
    save_checkpoint(out_dir / "actor_critic_droid_best.pt", {"model_state": model.state_dict(), "history": history})
    save_json(out_dir / "actor_critic_droid_history.json", {"history": history, "num_samples": len(samples)})
    print({"history": history, "num_samples": len(samples)})


if __name__ == "__main__":
    main()
