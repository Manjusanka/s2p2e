from __future__ import annotations

from pathlib import Path

from s2p2e.utils.io import save_json


def export_paper_tables(path: str | Path, summary: dict) -> None:
    tables = {
        "table_main": {
            "full": summary.get("joint", {}),
            "ablations": summary.get("ablations", {}),
        },
        "table_layer1": summary.get("layer1", {}),
        "table_layer2": summary.get("layer2", {}),
        "table_layer3": summary.get("layer3", {}),
    }
    save_json(path, tables)


def export_failure_analysis(path: str | Path, summary: dict) -> None:
    ablations = summary.get("ablations", {})
    full_loss = ablations.get("full", {}).get("loss", 0.0)
    rag_loss = ablations.get("w_o_rag", {}).get("loss", full_loss)
    ppe_loss = ablations.get("w_o_ppe", {}).get("loss", full_loss)
    residual_loss = ablations.get("w_o_residual", {}).get("loss", full_loss)
    payload = {
        "semantic_failures_proxy": max(rag_loss - full_loss, 0.0),
        "physics_failures_proxy": max(ppe_loss - full_loss, 0.0),
        "execution_failures_proxy": max(residual_loss - full_loss, 0.0),
    }
    save_json(path, payload)


def export_method_appendix(path: str | Path, config: dict) -> None:
    appendix = {
        "training_protocol": {
            "layer1": {
                "embedding_dim": config["model"]["embedding_dim"],
                "retrieval_topk": config["model"]["retrieval_topk"],
                "lambda_pose": config["training"]["lambda_pose"],
                "lambda_gripper": config["training"]["lambda_gripper"],
                "lambda_retrieval": config["training"]["lambda_retrieval"],
            },
            "layer2": {
                "ppe_hidden_dim": config["model"]["ppe_hidden_dim"],
                "ppe_layers": config["model"]["ppe_layers"],
                "lambda_physics": config["training"]["lambda_physics"],
                "trajectory_length": config["physics"]["trajectory_length"],
            },
            "layer3": {
                "history_length": config["physics"]["history_length"],
                "clip_ratio": config["model"]["control_clip_ratio"],
                "lambda_gate": config["training"]["lambda_gate"],
            },
        }
    }
    save_json(path, appendix)
