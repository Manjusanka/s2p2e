from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from s2p2e.models.encoders import PointNetEncoder


class OptionalExternalBackbone(nn.Module):
    def __init__(self, name: str, repo_path: str | Path | None, fallback: nn.Module) -> None:
        super().__init__()
        self.name = name
        self.repo_path = Path(repo_path) if repo_path else None
        self.fallback = fallback
        self.backend = "fallback"

    def available(self) -> bool:
        return bool(self.repo_path and self.repo_path.exists() and any(self.repo_path.iterdir()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fallback(x)


class PointNeXtAdapter(OptionalExternalBackbone):
    def __init__(self, repo_path: str | Path | None = None, output_dim: int = 512) -> None:
        super().__init__("pointnext", repo_path, PointNetEncoder(output_dim=output_dim))


class SAM2Adapter(OptionalExternalBackbone):
    def __init__(self, repo_path: str | Path | None = None) -> None:
        super().__init__("sam2", repo_path, nn.Identity())

    def segment(self, image: torch.Tensor) -> torch.Tensor:
        return image


class DepthAnythingV2Adapter(OptionalExternalBackbone):
    def __init__(self, repo_path: str | Path | None = None) -> None:
        super().__init__("depth_anything_v2", repo_path, nn.Identity())

    def predict_depth(self, image: torch.Tensor) -> torch.Tensor:
        return image.mean(dim=1, keepdim=True)
