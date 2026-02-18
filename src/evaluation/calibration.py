"""Calibration metrics and visualization helpers."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

__all__ = [
    "expected_calibration_error",
    "plot_reliability_diagram",
    "temperature_scaling",
]


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Compute expected calibration error for multi-class predictions."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        if not np.any(in_bin):
            continue
        bin_acc = accuracies[in_bin].mean()
        bin_conf = confidences[in_bin].mean()
        ece += np.abs(bin_acc - bin_conf) * in_bin.mean()
    return float(ece)


def plot_reliability_diagram(probs: np.ndarray, labels: np.ndarray, save_path: str | Path) -> None:
    """Plot and save reliability diagram."""
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    accuracies = (predictions == labels).astype(float)
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    bin_accs = []
    for i in range(len(bins) - 1):
        in_bin = (confidences > bins[i]) & (confidences <= bins[i + 1])
        bin_accs.append(accuracies[in_bin].mean() if np.any(in_bin) else 0.0)

    plt.figure(figsize=(5, 5))
    plt.plot([0, 1], [0, 1], "--", color="gray", label="Perfectly calibrated")
    plt.bar(bin_centers, bin_accs, width=0.09, alpha=0.7, label="Observed accuracy")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def temperature_scaling(logits: np.ndarray, labels: np.ndarray) -> float:
    """Estimate a scalar temperature minimizing NLL on validation logits."""
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    temperature = torch.nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([temperature], lr=0.1, max_iter=50)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(logits_t / temperature.clamp(min=1e-3), labels_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().clamp(min=1e-3).item())
