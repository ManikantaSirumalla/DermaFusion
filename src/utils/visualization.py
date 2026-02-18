"""Visualization helper utilities for training and evaluation analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import roc_curve, auc

__all__ = [
    "plot_ablation_comparison",
    "plot_class_distribution",
    "plot_confusion_matrix",
    "plot_roc_curves",
    "plot_training_curves",
]


def _ensure_parent(save_path: str | Path) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
    normalize: bool = True,
) -> None:
    """Plot confusion matrix and save image."""
    cm_plot = cm.astype(float)
    if normalize:
        row_sum = cm_plot.sum(axis=1, keepdims=True) + 1e-12
        cm_plot = cm_plot / row_sum

    plt.figure(figsize=(7, 6))
    sns.heatmap(cm_plot, annot=True, fmt=".2f" if normalize else "d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(np.arange(len(class_names)) + 0.5, class_names, rotation=45, ha="right")
    plt.yticks(np.arange(len(class_names)) + 0.5, class_names, rotation=0)
    plt.tight_layout()
    plt.savefig(_ensure_parent(save_path))
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """Plot one-vs-rest ROC curves for all classes."""
    plt.figure(figsize=(7, 6))
    for idx, name in enumerate(class_names):
        y_true_bin = (y_true == idx).astype(int)
        fpr, tpr, _ = roc_curve(y_true_bin, y_probs[:, idx])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest)")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(_ensure_parent(save_path))
    plt.close()


def plot_training_curves(
    train_log: dict[str, list[float]],
    val_log: dict[str, list[float]],
    save_path: str | Path,
) -> None:
    """Plot training and validation loss/metric curves."""
    epochs = np.arange(1, len(train_log.get("loss", [])) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_log.get("loss", []), label="train_loss")
    plt.plot(epochs, val_log.get("loss", []), label="val_loss")
    plt.legend()
    plt.title("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_log.get("balanced_accuracy", []), label="train_bal_acc")
    plt.plot(epochs, val_log.get("balanced_accuracy", []), label="val_bal_acc")
    plt.legend()
    plt.title("Balanced Accuracy")
    plt.tight_layout()
    plt.savefig(_ensure_parent(save_path))
    plt.close()


def plot_class_distribution(
    labels: np.ndarray,
    class_names: list[str],
    save_path: str | Path,
) -> None:
    """Plot class count bar chart."""
    counts = np.bincount(labels.astype(int), minlength=len(class_names))
    plt.figure(figsize=(8, 4))
    sns.barplot(x=class_names, y=counts, palette="viridis")
    plt.ylabel("Count")
    plt.title("Class Distribution")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(_ensure_parent(save_path))
    plt.close()


def plot_ablation_comparison(results_dict: dict[str, float], metric: str, save_path: str | Path) -> None:
    """Plot ablation experiment comparison for a chosen metric."""
    names = list(results_dict.keys())
    values = [results_dict[name] for name in names]
    plt.figure(figsize=(12, 4))
    sns.barplot(x=names, y=values, palette="magma")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.title(f"Ablation Comparison: {metric}")
    plt.tight_layout()
    plt.savefig(_ensure_parent(save_path))
    plt.close()
