"""Statistical comparison and confidence interval utilities."""

from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "bootstrap_confidence_intervals",
    "compute_all_cis",
    "mcnemar_test",
]


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for scalar metric."""
    rng = np.random.default_rng(42)
    n = len(y_true)
    values = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        values.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1.0 - ci) / 2.0
    return float(np.quantile(values, alpha)), float(np.quantile(values, 1.0 - alpha))


def mcnemar_test(model_a_preds: np.ndarray, model_b_preds: np.ndarray, labels: np.ndarray) -> float:
    """Approximate McNemar test p-value via chi-square statistic."""
    a_correct = model_a_preds == labels
    b_correct = model_b_preds == labels
    b = np.sum(a_correct & ~b_correct)
    c = np.sum(~a_correct & b_correct)
    stat = ((abs(b - c) - 1) ** 2) / (b + c + 1e-12)
    # One-dof chi-square approximation survival function.
    p_value = float(np.exp(-0.5 * stat))
    return p_value


def compute_all_cis(results_dict: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, tuple[float, float]]:
    """Compute bootstrap CIs for a dictionary of metric inputs."""
    output: dict[str, tuple[float, float]] = {}
    for metric_name, (y_true, y_pred) in results_dict.items():
        metric_fn = lambda yt, yp: float(np.mean(yt == yp))
        output[metric_name] = bootstrap_confidence_intervals(y_true, y_pred, metric_fn)
    return output
