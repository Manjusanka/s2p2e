from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from s2p2e.data.corpus import build_object_corpus
from s2p2e.data.datasets import (
    JointStackDataset,
    PhysicsTrajectoryDataset,
    ResidualSignalDataset,
    Layer1Dataset,
    build_tokenizer,
    collate_joint,
    collate_layer1,
    split_records,
)
from s2p2e.data.cache import PointCloudCache
from s2p2e.models.full_stack import FullS2P2E
from s2p2e.models.layer1 import Layer1Model
from s2p2e.models.layer2 import PhysicsPriorEncoder
from s2p2e.models.layer3 import ResidualController
from s2p2e.models.router import TaskWeightRouter
from s2p2e.trainers.ablations import export_ablation_results
from s2p2e.trainers.common import average_logs, move_to_device
from s2p2e.trainers.joint import eval_joint_epoch, train_joint_epoch
from s2p2e.trainers.paper_eval import evaluate_kb_tiers, evaluate_task_families
from s2p2e.trainers.reporting import export_failure_analysis, export_method_appendix, export_paper_tables
from s2p2e.utils.config import load_config
from s2p2e.utils.io import ensure_dir, save_checkpoint, save_json
from s2p2e.utils.seed import set_seed


def train_layer1(model: Layer1Model, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, cfg: dict) -> dict[str, float]:
    model.train()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(batch["points"], batch["tokens"], batch["geometric"], batch["operational"])
        loss, metrics = model.loss(
            outputs,
            batch["pose"],
            cfg["lambda_pose"],
            cfg["lambda_gripper"],
            cfg["lambda_retrieval"],
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logs.append({"loss": float(loss.item()), **metrics})
    return average_logs(logs)


@torch.no_grad()
def eval_layer1(model: Layer1Model, loader: DataLoader, device: torch.device, cfg: dict) -> dict[str, float]:
    model.eval()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(batch["points"], batch["tokens"], batch["geometric"], batch["operational"])
        loss, metrics = model.loss(
            outputs,
            batch["pose"],
            cfg["lambda_pose"],
            cfg["lambda_gripper"],
            cfg["lambda_retrieval"],
        )
        pose_mae = (outputs["poses"][:, 0, :3] - batch["pose"][:, :3]).abs().mean().item()
        logs.append({"loss": float(loss.item()), "pose_mae": pose_mae, **metrics})
    return average_logs(logs)


def train_layer2(model: PhysicsPriorEncoder, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, cfg: dict) -> dict[str, float]:
    model.train()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        state = batch["state"].reshape(-1, batch["state"].shape[-1])
        tau = batch["tau"].reshape(-1, batch["tau"].shape[-1])
        embodiment = batch["embodiment"].unsqueeze(1).expand(-1, batch["state"].shape[1], -1).reshape(state.shape[0], -1)
        outputs = model(state, embodiment)
        loss, metrics = model.loss(outputs, tau, cfg["torque_limit"], cfg["lambda_physics"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logs.append({"loss": float(loss.item()), **metrics})
    return average_logs(logs)


@torch.no_grad()
def eval_layer2(model: PhysicsPriorEncoder, loader: DataLoader, device: torch.device, cfg: dict) -> dict[str, float]:
    model.eval()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        state = batch["state"].reshape(-1, batch["state"].shape[-1])
        tau = batch["tau"].reshape(-1, batch["tau"].shape[-1])
        embodiment = batch["embodiment"].unsqueeze(1).expand(-1, batch["state"].shape[1], -1).reshape(state.shape[0], -1)
        outputs = model(state, embodiment)
        loss, metrics = model.loss(outputs, tau, cfg["torque_limit"], cfg["lambda_physics"])
        rmse = torch.sqrt(torch.mean((outputs["tau"] - tau) ** 2)).item()
        logs.append({"loss": float(loss.item()), "rmse": rmse, **metrics})
    return average_logs(logs)


def train_layer3(model: ResidualController, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device, cfg: dict) -> dict[str, float]:
    model.train()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["qd_hist"],
            batch["tau_hist"],
            batch["pose_hist"],
            batch["force"],
            batch["pose_cur"],
            batch["phase"],
            batch["tau_pid"],
        )
        loss, metrics = model.loss(outputs, batch["target_residual"], batch["target_alpha"], cfg["lambda_gate"])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        logs.append({"loss": float(loss.item()), **metrics})
    return average_logs(logs)


@torch.no_grad()
def eval_layer3(model: ResidualController, loader: DataLoader, device: torch.device, cfg: dict) -> dict[str, float]:
    model.eval()
    logs = []
    for batch in loader:
        batch = move_to_device(batch, device)
        outputs = model(
            batch["qd_hist"],
            batch["tau_hist"],
            batch["pose_hist"],
            batch["force"],
            batch["pose_cur"],
            batch["phase"],
            batch["tau_pid"],
        )
        loss, metrics = model.loss(outputs, batch["target_residual"], batch["target_alpha"], cfg["lambda_gate"])
        command_norm = outputs["command"].norm(dim=-1).mean().item()
        logs.append({"loss": float(loss.item()), "command_norm": command_norm, **metrics})
    return average_logs(logs)


def run_with_best(
    name: str,
    epochs: int,
    train_fn,
    eval_fn,
    model,
    train_loader,
    val_loader,
    optimizer,
    device,
    cfg,
    checkpoint_dir: Path,
) -> dict[str, float]:
    best_metrics = None
    best_loss = float("inf")
    history = []
    for epoch in range(epochs):
        train_metrics = train_fn(model, train_loader, optimizer, device, cfg)
        val_metrics = eval_fn(model, val_loader, device, cfg)
        record = {"epoch": epoch + 1, "train": train_metrics, "val": val_metrics}
        history.append(record)
        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_metrics = record
            save_checkpoint(
                checkpoint_dir / f"{name}_best.pt",
                {"model_state": model.state_dict(), "record": record},
            )
    save_json(checkpoint_dir / f"{name}_history.json", {"history": history, "best": best_metrics})
    return best_metrics or {}


def run_pipeline(cfg: dict) -> dict:
    set_seed(cfg["seed"])

    run_dir = ensure_dir(Path(cfg["artifacts_dir"]) / cfg["run_name"])
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    metric_dir = ensure_dir(run_dir / "metrics")

    data_cfg = cfg["data"]
    train_cfg = cfg["training"]
    model_cfg = cfg["model"]
    physics_cfg = cfg["physics"]
    device = torch.device(train_cfg["device"])

    shapenet_roots = data_cfg.get("shapenet_roots", data_cfg.get("shapenet_root"))
    records = build_object_corpus(
        shapenet_roots,
        data_cfg["objaverse_root"],
        data_cfg["shapenet_limit_per_category"],
        data_cfg["objaverse_limit"],
    )
    splits = split_records(records, data_cfg["train_ratio"], data_cfg["val_ratio"])
    tokenizer = build_tokenizer(records, data_cfg["tasks"])
    cache = PointCloudCache(data_cfg.get("cache_root", Path(cfg["artifacts_dir"]) / "cache"), data_cfg["num_points"])

    layer1_train = Layer1Dataset(splits["train"], data_cfg["tasks"], tokenizer, data_cfg["num_points"], "train", cache=cache)
    layer1_val = Layer1Dataset(splits["val"] or splits["train"][:1], data_cfg["tasks"], tokenizer, data_cfg["num_points"], "val", cache=cache)
    loader_kwargs = dict(batch_size=train_cfg["batch_size"], num_workers=train_cfg["num_workers"])
    layer1_train_loader = DataLoader(layer1_train, shuffle=True, collate_fn=collate_layer1, **loader_kwargs)
    layer1_val_loader = DataLoader(layer1_val, shuffle=False, collate_fn=collate_layer1, **loader_kwargs)

    layer1 = Layer1Model(tokenizer.vocab_size, model_cfg["embedding_dim"], model_cfg["retrieval_topk"]).to(device)
    layer1_opt = torch.optim.AdamW(layer1.parameters(), lr=train_cfg["lr_layer1"], weight_decay=train_cfg["weight_decay"])
    layer1_best = run_with_best(
        "layer1",
        train_cfg["epochs_layer1"],
        train_layer1,
        eval_layer1,
        layer1,
        layer1_train_loader,
        layer1_val_loader,
        layer1_opt,
        device,
        train_cfg,
        checkpoint_dir,
    )

    traj_train = PhysicsTrajectoryDataset(splits["train"], physics_cfg["num_trajectories"], physics_cfg["trajectory_length"], physics_cfg["dof"])
    traj_val = PhysicsTrajectoryDataset(splits["val"] or splits["train"][:1], max(16, physics_cfg["num_trajectories"] // 8), physics_cfg["trajectory_length"], physics_cfg["dof"])
    traj_train_loader = DataLoader(traj_train, shuffle=True, **loader_kwargs)
    traj_val_loader = DataLoader(traj_val, shuffle=False, **loader_kwargs)

    layer2 = PhysicsPriorEncoder(
        state_dim=physics_cfg["dof"] * 3,
        cond_dim=physics_cfg["dof"] * 3,
        dof=physics_cfg["dof"],
        hidden_dim=model_cfg["ppe_hidden_dim"],
        layers=model_cfg["ppe_layers"],
    ).to(device)
    layer2_opt = torch.optim.AdamW(layer2.parameters(), lr=train_cfg["lr_layer2"], weight_decay=train_cfg["weight_decay"])
    layer2_best = run_with_best(
        "layer2",
        train_cfg["epochs_layer2"],
        train_layer2,
        eval_layer2,
        layer2,
        traj_train_loader,
        traj_val_loader,
        layer2_opt,
        device,
        {"torque_limit": physics_cfg["torque_limit"], "lambda_physics": train_cfg["lambda_physics"]},
        checkpoint_dir,
    )

    resid_train = ResidualSignalDataset(physics_cfg["num_trajectories"], physics_cfg["history_length"], physics_cfg["dof"])
    resid_val = ResidualSignalDataset(max(16, physics_cfg["num_trajectories"] // 8), physics_cfg["history_length"], physics_cfg["dof"])
    resid_train_loader = DataLoader(resid_train, shuffle=True, **loader_kwargs)
    resid_val_loader = DataLoader(resid_val, shuffle=False, **loader_kwargs)

    layer3 = ResidualController(physics_cfg["dof"], physics_cfg["history_length"], model_cfg["control_clip_ratio"]).to(device)
    layer3_opt = torch.optim.AdamW(layer3.parameters(), lr=train_cfg["lr_layer3"], weight_decay=train_cfg["weight_decay"])
    layer3_best = run_with_best(
        "layer3",
        train_cfg["epochs_layer3"],
        train_layer3,
        eval_layer3,
        layer3,
        resid_train_loader,
        resid_val_loader,
        layer3_opt,
        device,
        {"lambda_gate": train_cfg["lambda_gate"]},
        checkpoint_dir,
    )

    joint_train = JointStackDataset(
        splits["train"],
        data_cfg["tasks"],
        tokenizer,
        data_cfg["num_points"],
        physics_cfg["history_length"],
        physics_cfg["dof"],
        cache=cache,
    )
    joint_val = JointStackDataset(
        splits["val"] or splits["train"][:1],
        data_cfg["tasks"],
        tokenizer,
        data_cfg["num_points"],
        physics_cfg["history_length"],
        physics_cfg["dof"],
        cache=cache,
    )
    joint_train_loader = DataLoader(joint_train, shuffle=True, collate_fn=collate_joint, **loader_kwargs)
    joint_val_loader = DataLoader(joint_val, shuffle=False, collate_fn=collate_joint, **loader_kwargs)

    router = TaskWeightRouter(model_cfg["task_weights"], data_cfg["tasks"]).to(device)
    full_model = FullS2P2E(layer1, layer2, layer3, router, dof=physics_cfg["dof"]).to(device)
    joint_opt = torch.optim.AdamW(full_model.parameters(), lr=train_cfg.get("lr_joint", 5e-4), weight_decay=train_cfg["weight_decay"])
    joint_best = run_with_best(
        "joint",
        train_cfg.get("epochs_joint", 1),
        train_joint_epoch,
        eval_joint_epoch,
        full_model,
        joint_train_loader,
        joint_val_loader,
        joint_opt,
        device,
        {
            "lambda_joint_pose": train_cfg.get("lambda_joint_pose", 3.0),
            "lambda_joint_tau": train_cfg.get("lambda_joint_tau", 1.0),
            "lambda_joint_feasibility": train_cfg.get("lambda_joint_feasibility", 0.5),
        },
        checkpoint_dir,
    )

    ablations = {
        "full": {},
        "w_o_rag": {"disable_rag": True, "disable_ppe": False, "disable_residual": False},
        "w_o_ppe": {"disable_rag": False, "disable_ppe": True, "disable_residual": False},
        "w_o_residual": {"disable_rag": False, "disable_ppe": False, "disable_residual": True},
    }
    ablation_results = {}
    for name, flags in ablations.items():
        full_model.set_ablation(
            flags.get("disable_rag", False),
            flags.get("disable_ppe", False),
            flags.get("disable_residual", False),
        )
        ablation_results[name] = eval_joint_epoch(
            full_model,
            joint_val_loader,
            device,
            {
                "lambda_joint_pose": train_cfg.get("lambda_joint_pose", 3.0),
                "lambda_joint_tau": train_cfg.get("lambda_joint_tau", 1.0),
                "lambda_joint_feasibility": train_cfg.get("lambda_joint_feasibility", 0.5),
            },
        )
    full_model.set_ablation(False, False, False)
    export_ablation_results(metric_dir / "ablations.json", ablation_results)
    kb_tier_results = evaluate_kb_tiers(layer1, layer1_val_loader, device, train_cfg)
    task_family_results = evaluate_task_families(
        full_model,
        joint_val_loader,
        device,
        {
            "lambda_joint_pose": train_cfg.get("lambda_joint_pose", 3.0),
            "lambda_joint_tau": train_cfg.get("lambda_joint_tau", 1.0),
            "lambda_joint_feasibility": train_cfg.get("lambda_joint_feasibility", 0.5),
        },
        data_cfg["tasks"],
    )

    summary = {
        "layer1": layer1_best,
        "layer2": layer2_best,
        "layer3": layer3_best,
        "joint": joint_best,
        "ablations": ablation_results,
        "kb_tiers": kb_tier_results,
        "task_families": task_family_results,
        "num_records": len(records),
    }
    save_json(metric_dir / "summary.json", summary)
    export_paper_tables(metric_dir / "paper_tables.json", summary)
    export_failure_analysis(metric_dir / "failure_analysis.json", summary)
    export_method_appendix(metric_dir / "method_appendix.json", cfg)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiment_small.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    summary = run_pipeline(cfg)
    print(summary)
