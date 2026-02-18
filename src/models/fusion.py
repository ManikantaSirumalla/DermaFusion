"""Fusion modules for multi-modal image + metadata representations."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["CrossAttentionFusion", "FiLMFusion", "LateFusion"]


class LateFusion(nn.Module):
    """Concatenate image and metadata features followed by MLP."""

    def __init__(self, image_dim: int, metadata_dim: int, hidden_dim: int = 512) -> None:
        """Initialize late fusion block."""
        super().__init__()
        self.fusion = nn.Sequential(
            nn.LayerNorm(image_dim + metadata_dim),
            nn.Linear(image_dim + metadata_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.output_dim = hidden_dim

    def forward(self, image_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        """Fuse image and metadata vectors."""
        combined = torch.cat([image_features, metadata_features], dim=1)
        return self.fusion(combined)


class FiLMFusion(nn.Module):
    """Feature-wise linear modulation of image features by metadata."""

    def __init__(self, image_dim: int, metadata_dim: int) -> None:
        """Initialize FiLM projections."""
        super().__init__()
        self.gamma = nn.Linear(metadata_dim, image_dim)
        self.beta = nn.Linear(metadata_dim, image_dim)
        self.output_dim = image_dim

    def forward(self, image_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        """Apply FiLM modulation."""
        gamma = self.gamma(metadata_features)
        beta = self.beta(metadata_features)
        return gamma * image_features + beta


class CrossAttentionFusion(nn.Module):
    """Cross-attention with metadata query and image key/value."""

    def __init__(self, image_dim: int, metadata_dim: int, num_heads: int = 4) -> None:
        """Initialize cross-attention fusion module."""
        super().__init__()
        self.meta_proj = nn.Linear(metadata_dim, image_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=image_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.output_dim = image_dim

    def forward(self, image_features: torch.Tensor, metadata_features: torch.Tensor) -> torch.Tensor:
        """Run one-step cross attention and return fused vector."""
        query = self.meta_proj(metadata_features).unsqueeze(1)
        key_value = image_features.unsqueeze(1)
        attended, _ = self.attn(query=query, key=key_value, value=key_value)
        return attended.squeeze(1)
