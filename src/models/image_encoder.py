"""Image backbone encoder built on top of timm models."""

from __future__ import annotations

from typing import Any

import timm
import torch
from torch import nn

__all__ = ["ImageEncoder", "resolve_backbone_name"]

_BACKBONE_MAP = {
    "efficientnet_b4": "efficientnet_b4",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "swin_tiny_patch4_window7_224": "swin_tiny_patch4_window7_224",
    "convnext_v2": "convnextv2_base",
    "convnextv2_base": "convnextv2_base",
}


def resolve_backbone_name(backbone_name: str) -> str:
    """Resolve project shorthand names to timm backbone names."""
    if backbone_name not in _BACKBONE_MAP:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    return _BACKBONE_MAP[backbone_name]


class ImageEncoder(nn.Module):
    """Configurable image feature encoder using timm."""

    def __init__(
        self,
        backbone_name: str,
        pretrained: bool = True,
        freeze: bool = False,
        output_dim: int | None = None,
    ) -> None:
        """Initialize timm feature extractor."""
        super().__init__()
        timm_name = resolve_backbone_name(backbone_name)
        self.backbone = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        self.feature_dim = int(getattr(self.backbone, "num_features"))
        self.output_dim = output_dim or self.feature_dim
        self.projection = (
            nn.Identity()
            if self.output_dim == self.feature_dim
            else nn.Linear(self.feature_dim, self.output_dim)
        )
        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract pooled image features with optional projection."""
        features = self.backbone(x)
        return self.projection(features)

    def unfreeze(self, num_layers: int = -1) -> None:
        """Unfreeze last N backbone parameter tensors, or all if -1."""
        params = list(self.backbone.parameters())
        if num_layers == -1 or num_layers >= len(params):
            for param in params:
                param.requires_grad = True
            return
        for param in params[-num_layers:]:
            param.requires_grad = True

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Return intermediate feature map representation for GradCAM use."""
        if hasattr(self.backbone, "forward_features"):
            feature_maps: Any = self.backbone.forward_features(x)
            if isinstance(feature_maps, (list, tuple)):
                return feature_maps[-1]
            return feature_maps
        return self.backbone(x).unsqueeze(-1).unsqueeze(-1)
