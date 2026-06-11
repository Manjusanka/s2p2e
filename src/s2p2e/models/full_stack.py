from __future__ import annotations

import torch
from torch import nn

from s2p2e.models.feasibility import FeasibilityFilter
from s2p2e.models.ik import PoseToJointIK
from s2p2e.models.layer1 import Layer1Model
from s2p2e.models.layer2 import PhysicsPriorEncoder
from s2p2e.models.layer3 import ResidualController
from s2p2e.models.policy_bridge import PPEPolicyBridge
from s2p2e.models.router import TaskWeightRouter


class FullS2P2E(nn.Module):
    def __init__(
        self,
        layer1: Layer1Model,
        layer2: PhysicsPriorEncoder,
        layer3: ResidualController,
        task_router: TaskWeightRouter,
        dof: int = 7,
    ) -> None:
        super().__init__()
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer3 = layer3
        self.task_router = task_router
        self.pose_to_joint = PoseToJointIK(pose_dim=8, dof=dof)
        self.feasibility_filter = FeasibilityFilter(8)
        ppe_feature_dim = layer2.output.in_features
        self.policy_bridge = PPEPolicyBridge(obs_dim=21, action_dim=7, ppe_feature_dim=ppe_feature_dim)
        self.dof = dof
        self.disable_rag = False
        self.disable_ppe = False
        self.disable_residual = False

    def set_ablation(self, disable_rag: bool, disable_ppe: bool, disable_residual: bool) -> None:
        self.disable_rag = disable_rag
        self.disable_ppe = disable_ppe
        self.disable_residual = disable_residual

    def forward(
        self,
        points: torch.Tensor,
        tokens: torch.Tensor,
        task_ids: torch.Tensor,
        geometric: torch.Tensor,
        operational: torch.Tensor,
        embodiment: torch.Tensor,
        qd_hist: torch.Tensor,
        tau_hist: torch.Tensor,
        pose_hist: torch.Tensor,
        force: torch.Tensor,
        pose_cur: torch.Tensor,
        phase: torch.Tensor,
        tau_pid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        layer1_out = self.layer1(points, tokens, geometric, operational)
        if self.disable_rag:
            layer1_out["poses"] = torch.zeros_like(layer1_out["poses"])
        task_weights = self.task_router(task_ids)
        pose_candidates = layer1_out["poses"]
        feasibility = self.feasibility_filter(pose_candidates[:, 0])
        ik = self.pose_to_joint(pose_candidates[:, 0])
        q_ref = ik["q_ref"]
        qd_ref = ik["qd_ref"]
        qdd_ref = ik["qdd_ref"]
        state = torch.cat([q_ref, qd_ref, qdd_ref], dim=-1)
        ppe_out = self.layer2(state, embodiment)
        if self.disable_ppe:
            ppe_out["tau"] = torch.zeros_like(ppe_out["tau"])
            ppe_out["feasible"] = torch.zeros_like(ppe_out["feasible"]) + 0.5
        controller_out = self.layer3(qd_hist, tau_hist, pose_hist, force, pose_cur, phase, tau_pid)
        if self.disable_residual:
            controller_out["residual"] = torch.zeros_like(controller_out["residual"])
            controller_out["clipped"] = torch.zeros_like(controller_out["clipped"])
            controller_out["command"] = tau_pid
        policy_action = self.policy_bridge.act(state, ppe_out["features"])
        return {
            "layer1": layer1_out,
            "task_weights": task_weights,
            "feasibility": feasibility,
            "ik": ik,
            "q_ref": q_ref,
            "ppe": ppe_out,
            "controller": controller_out,
            "policy_action": policy_action,
        }

    @staticmethod
    def joint_loss(
        outputs: dict[str, torch.Tensor],
        target_pose: torch.Tensor,
        target_tau: torch.Tensor,
        lambda_joint_pose: float,
        lambda_joint_tau: float,
        lambda_joint_feasibility: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pose_loss = torch.nn.functional.mse_loss(outputs["layer1"]["poses"][:, 0], target_pose)
        tau_loss = torch.nn.functional.mse_loss(outputs["ppe"]["tau"], target_tau)
        feasible_target = (target_tau.abs().max(dim=-1, keepdim=True).values < 12.0).float()
        feasibility_loss = torch.nn.functional.binary_cross_entropy(outputs["ppe"]["feasible"], feasible_target)
        total = (
            lambda_joint_pose * pose_loss
            + lambda_joint_tau * tau_loss
            + lambda_joint_feasibility * feasibility_loss
        )
        return total, {
            "joint_pose_loss": float(pose_loss.item()),
            "joint_tau_loss": float(tau_loss.item()),
            "joint_feasibility_loss": float(feasibility_loss.item()),
        }
