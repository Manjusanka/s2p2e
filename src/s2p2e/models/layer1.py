from __future__ import annotations

import faiss
import torch
import torch.nn.functional as F
from torch import nn

from s2p2e.models.encoders import MLPProjector, PointNetEncoder, TextEncoder
from s2p2e.models.backbones import PointNeXtAdapter
from s2p2e.models.grasp_proposal import GraspPoseProposalNetwork


class Layer1Model(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 512, retrieval_topk: int = 5) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.retrieval_topk = retrieval_topk
        self.point_encoder = PointNeXtAdapter(output_dim=embedding_dim)
        self.text_encoder = TextEncoder(vocab_size=vocab_size, output_dim=embedding_dim)
        self.geom_encoder = MLPProjector(12, embedding_dim)
        self.op_encoder = MLPProjector(12, embedding_dim)
        self.query_fuser = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.ReLU(),
        )
        self.cross_attn = nn.MultiheadAttention(embedding_dim, num_heads=8, batch_first=True)
        self.pose_head = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, retrieval_topk * 8),
        )
        self.grasp_proposal = GraspPoseProposalNetwork(embedding_dim=embedding_dim, num_candidates=retrieval_topk)
        self.semantic_enabled = True
        self.geometric_enabled = True
        self.operational_enabled = True

    def set_kb_tiers(self, semantic: bool, geometric: bool, operational: bool) -> None:
        self.semantic_enabled = semantic
        self.geometric_enabled = geometric
        self.operational_enabled = operational

    def forward(
        self,
        points: torch.Tensor,
        tokens: torch.Tensor,
        geometric: torch.Tensor,
        operational: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        scene = self.point_encoder(points)
        semantic = self.text_encoder(tokens)
        geom = self.geom_encoder(geometric)
        ops = self.op_encoder(operational)
        query = self.query_fuser(torch.cat([scene, semantic], dim=-1))
        semantic_mem = semantic if self.semantic_enabled else torch.zeros_like(semantic)
        geom_mem = geom if self.geometric_enabled else torch.zeros_like(geom)
        ops_mem = ops if self.operational_enabled else torch.zeros_like(ops)
        memory = torch.stack([semantic_mem, geom_mem, ops_mem], dim=1)
        attn_out, attn_weights = self.cross_attn(query.unsqueeze(1), memory, memory)
        fused = torch.cat([query, attn_out.squeeze(1)], dim=-1)
        raw_poses = self.pose_head(fused).view(points.shape[0], self.retrieval_topk, 8)
        poses = torch.cat(
            [
                raw_poses[..., :3],
                F.normalize(raw_poses[..., 3:7], dim=-1),
                raw_poses[..., 7:8],
            ],
            dim=-1,
        )
        proposal = self.grasp_proposal(scene, geom_mem, ops_mem)
        combined_scores = proposal["scores"] + torch.softmax(attn_weights.squeeze(1), dim=-1).mean(dim=-1, keepdim=True)
        tier_count = max(int(self.semantic_enabled) + int(self.geometric_enabled) + int(self.operational_enabled), 1)
        retrieval_bank = F.normalize((semantic_mem + geom_mem + ops_mem) / float(tier_count), dim=-1)
        query_norm = F.normalize(query, dim=-1)
        retrieval_logits = query_norm @ retrieval_bank.t()
        return {
            "query": query,
            "retrieval_logits": retrieval_logits,
            "poses": poses,
            "proposal": proposal,
            "combined_scores": combined_scores,
            "attention": attn_weights,
            "bank": retrieval_bank,
        }

    @staticmethod
    def loss(
        outputs: dict[str, torch.Tensor],
        targets: torch.Tensor,
        lambda_pose: float,
        lambda_gripper: float,
        lambda_retrieval: float,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        pred_pose = outputs["poses"][:, 0, :7]
        pred_gripper = outputs["poses"][:, 0, 7]
        pose_loss = F.mse_loss(pred_pose, targets[:, :7])
        gripper_loss = F.binary_cross_entropy_with_logits(pred_gripper, targets[:, 7])
        labels = torch.arange(targets.shape[0], device=targets.device)
        retrieval_loss = F.cross_entropy(outputs["retrieval_logits"], labels)
        total = lambda_pose * pose_loss + lambda_gripper * gripper_loss + lambda_retrieval * retrieval_loss
        return total, {
            "pose_loss": float(pose_loss.item()),
            "gripper_loss": float(gripper_loss.item()),
            "retrieval_loss": float(retrieval_loss.item()),
        }

    def build_faiss_index(self, embeddings: torch.Tensor) -> faiss.IndexFlatIP:
        bank = F.normalize(embeddings, dim=-1).detach().cpu().numpy().astype("float32")
        index = faiss.IndexFlatIP(bank.shape[1])
        index.add(bank)
        return index
