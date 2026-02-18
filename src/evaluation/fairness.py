"""Fairness metrics for demographic subgroup analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "demographic_parity",
    "equalized_odds",
    "per_group_performance",
]


def demographic_parity(predictions: np.ndarray, labels: np.ndarray, sensitive_attr: np.ndarray) -> dict:
    """Compute positive-rate parity across sensitive groups."""
    _ = labels
    result: dict[str, float] = {}
    for group in np.unique(sensitive_attr):
        mask = sensitive_attr == group
        result[str(group)] = float(np.mean(predictions[mask]))
    return result


def equalized_odds(predictions: np.ndarray, labels: np.ndarray, sensitive_attr: np.ndarray) -> dict:
    """Compute TPR/FPR per sensitive group for binary view of predictions."""
    result: dict[str, dict[str, float]] = {}
    for group in np.unique(sensitive_attr):
        mask = sensitive_attr == group
        y_true = labels[mask]
        y_pred = predictions[mask]
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        result[str(group)] = {"tpr": float(tpr), "fpr": float(fpr)}
    return result


def per_group_performance(predictions: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> pd.DataFrame:
    """Build per-group accuracy summary."""
    rows = []
    for group in np.unique(groups):
        mask = groups == group
        acc = float(np.mean(predictions[mask] == labels[mask]))
        rows.append({"group": group, "count": int(mask.sum()), "accuracy": acc})
    return pd.DataFrame(rows)
