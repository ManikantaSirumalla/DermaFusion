"""Loss functions for imbalanced skin cancer classification."""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["CombinedLoss", "CostSensitiveLoss", "FocalLoss"]


class FocalLoss(nn.Module):
    """Focal loss for class-imbalanced multi-class classification."""

    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        reduction: str = "mean",
    ) -> None:
        """Initialize focal loss parameters."""
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss."""
        ce = F.cross_entropy(logits, targets, reduction="none")
        p_t = torch.exp(-ce)
        focal = (1.0 - p_t) ** self.gamma * ce

        if self.alpha is not None:
            alpha = self.alpha.to(logits.device, dtype=logits.dtype)
            focal = alpha[targets] * focal

        if self.reduction == "mean":
            return focal.mean()
        if self.reduction == "sum":
            return focal.sum()
        return focal


class CostSensitiveLoss(nn.Module):
    """Cost-weighted cross-entropy emphasizing clinical false negatives."""

    _CLASS_TO_INDEX = {"mel": 0, "nv": 1, "bcc": 2, "akiec": 3, "bkl": 4, "df": 5, "vasc": 6}

    def __init__(self, cost_matrix: Mapping[str, Mapping[str, float]], num_classes: int = 7) -> None:
        """Initialize with class-level FN weights."""
        super().__init__()
        weights = torch.ones(num_classes, dtype=torch.float32)
        default_fn = float(cost_matrix.get("default", {}).get("fn_weight", 1.0))
        weights *= default_fn
        for class_name, class_idx in self._CLASS_TO_INDEX.items():
            if class_name in cost_matrix:
                weights[class_idx] = float(cost_matrix[class_name].get("fn_weight", default_fn))
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy."""
        return F.cross_entropy(logits, targets, weight=self.class_weights.to(logits.device))


class CombinedLoss(nn.Module):
    """Weighted combination of focal and label-smoothed cross entropy."""

    def __init__(
        self,
        focal_weight: float = 0.7,
        ce_weight: float = 0.3,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.1,
    ) -> None:
        """Initialize combined loss parameters."""
        super().__init__()
        self.focal_weight = focal_weight
        self.ce_weight = ce_weight
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma, reduction="mean")
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute combined focal + CE loss."""
        focal = self.focal_loss(logits, targets)
        ce = F.cross_entropy(logits, targets, label_smoothing=self.label_smoothing)
        return self.focal_weight * focal + self.ce_weight * ce
