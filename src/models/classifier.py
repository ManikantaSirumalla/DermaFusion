"""Classification head for fused features."""

from __future__ import annotations

from torch import nn

__all__ = ["ClassificationHead"]


class ClassificationHead(nn.Module):
    """Final MLP classification head returning logits."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int = 7,
        hidden_dim: int = 512,
        dropout: float = 0.4,
    ) -> None:
        """Initialize classification head."""
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        """Return raw logits."""
        return self.classifier(x)
