"""Step 6 evaluation framework tests."""

from __future__ import annotations

from pathlib import Path
import tempfile

import numpy as np
import torch
from torch import nn

from src.evaluation.calibration import expected_calibration_error, temperature_scaling
from src.evaluation.interpretability import generate_gradcam
from src.evaluation.metrics import MetricCalculator
from src.evaluation.statistical import bootstrap_confidence_intervals


class TinyImageModel(nn.Module):
    """Small image-only model for interpretability tests."""

    def __init__(self, num_classes: int = 7) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Flatten(), nn.Linear(3 * 16 * 16, num_classes))

    def forward(self, batch):
        return self.net(batch["image"])


def test_metric_calculator_outputs_expected_keys() -> None:
    """Metric calculator should return required summary keys."""
    calc = MetricCalculator(num_classes=3, class_names=["a", "b", "c"])
    y_true = np.array([0, 1, 2, 1])
    y_pred = np.array([0, 1, 1, 1])
    y_prob = np.array(
        [
            [0.9, 0.05, 0.05],
            [0.1, 0.8, 0.1],
            [0.2, 0.5, 0.3],
            [0.2, 0.6, 0.2],
        ]
    )
    calc.update(y_pred, y_true, y_prob)
    out = calc.compute()
    assert "balanced_accuracy" in out
    assert "macro_f1" in out
    assert "per_class" in out
    assert out["confusion_matrix"].shape == (3, 3)


def test_ece_in_valid_range() -> None:
    """ECE should be in [0, 1]."""
    probs = np.array([[0.8, 0.2], [0.3, 0.7], [0.55, 0.45]])
    labels = np.array([0, 1, 1])
    ece = expected_calibration_error(probs, labels, n_bins=5)
    assert 0.0 <= ece <= 1.0


def test_temperature_scaling_positive() -> None:
    """Estimated temperature should be positive."""
    logits = np.array([[3.0, 1.0], [0.5, 2.0], [1.2, 1.1]])
    labels = np.array([0, 1, 1])
    temp = temperature_scaling(logits, labels)
    assert temp > 0.0


def test_bootstrap_ci_range_ordering() -> None:
    """Bootstrap CI lower bound should not exceed upper bound."""
    y_true = np.array([0, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 1, 0, 0, 1, 1])
    metric_fn = lambda yt, yp: float(np.mean(yt == yp))
    low, high = bootstrap_confidence_intervals(y_true, y_pred, metric_fn, n_bootstrap=200)
    assert low <= high


def test_gradcam_returns_expected_shape() -> None:
    """GradCAM helper should return a 2D heatmap with image spatial size."""
    model = TinyImageModel(num_classes=7)
    image = torch.randn(3, 16, 16, requires_grad=True)
    heatmap = generate_gradcam(model, image, target_class=1, layer=None)
    assert heatmap.shape == (16, 16)
    assert np.isfinite(heatmap).all()
