from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class GraspPoseProposalNetwork(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_candidates: int = 16) -> None:
        super().__init__()
        self.num_candidates = num_candidates
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 3, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
        )
        self.pose_head = nn.Linear(embedding_dim, num_candidates * 8)
        self.score_head = nn.Linear(embedding_dim, num_candidates)
        self.width_head = nn.Linear(embedding_dim, num_candidates)

    def forward(
        self,
        scene_embed: torch.Tensor,
        geom_embed: torch.Tensor,
        op_embed: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        fused = self.fusion(torch.cat([scene_embed, geom_embed, op_embed], dim=-1))
        raw_poses = self.pose_head(fused).view(scene_embed.shape[0], self.num_candidates, 8)
        poses = torch.cat(
            [raw_poses[..., :3], F.normalize(raw_poses[..., 3:7], dim=-1), raw_poses[..., 7:8]],
            dim=-1,
        )
        scores = self.score_head(fused)
        widths = torch.sigmoid(self.width_head(fused))
        return {"poses": poses, "scores": scores, "widths": widths}

    @staticmethod
    def loss(outputs: dict[str, torch.Tensor], target_pose: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        scores = outputs["scores"]
        best_idx = scores.argmax(dim=-1)
        pred_pose = outputs["poses"][torch.arange(scores.shape[0], device=scores.device), best_idx]
        pose_loss = F.mse_loss(pred_pose, target_pose)
        score_reg = -torch.softmax(scores, dim=-1).max(dim=-1).values.mean()
        total = pose_loss + 0.05 * score_reg
        return total, {"proposal_pose_loss": float(pose_loss.item()), "proposal_score_reg": float(score_reg.item())}
