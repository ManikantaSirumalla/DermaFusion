"""Clinical metadata encoder with learned embeddings."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["MetadataEncoder"]


class MetadataEncoder(nn.Module):
    """Encode structured metadata into a dense feature vector."""

    def __init__(
        self,
        num_sex_categories: int,
        num_loc_categories: int,
        embedding_dim: int = 32,
        output_dim: int = 128,
    ) -> None:
        """Initialize metadata embedding/projection layers."""
        super().__init__()
        self.sex_embedding = nn.Embedding(num_sex_categories, embedding_dim)
        self.localization_embedding = nn.Embedding(num_loc_categories, embedding_dim)
        self.age_projection = nn.Linear(1, embedding_dim)
        self.missing_projection = nn.Linear(3, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim * 4, output_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(output_dim, output_dim),
        )
        self.output_dim = output_dim

    def forward(self, metadata: torch.Tensor) -> torch.Tensor:
        """Encode metadata tensor into dense features.

        Expected input layout per row:
        [age_norm, sex_idx, localization_idx, age_missing, sex_missing, localization_missing]
        """
        age = metadata[:, 0:1].float()
        sex_idx = metadata[:, 1].long()
        loc_idx = metadata[:, 2].long()
        missing_flags = metadata[:, 3:6].float()

        age_feat = self.age_projection(age)
        sex_feat = self.sex_embedding(sex_idx)
        loc_feat = self.localization_embedding(loc_idx)
        miss_feat = self.missing_projection(missing_flags)

        features = torch.cat([age_feat, sex_feat, loc_feat, miss_feat], dim=1)
        return self.mlp(features)
