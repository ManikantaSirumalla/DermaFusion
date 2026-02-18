"""Comprehensive evaluation metrics for skin cancer classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.metrics import auc as sklearn_auc
from sklearn.metrics import balanced_accuracy_score

__all__ = ["MetricCalculator"]


@dataclass
class MetricCalculator:
    """Accumulate predictions and compute classification metrics."""

    num_classes: int
    class_names: Sequence[str]
    y_true: list[int] = field(default_factory=list)
    y_pred: list[int] = field(default_factory=list)
    y_prob: list[np.ndarray] = field(default_factory=list)

    def update(
        self, predictions: np.ndarray, labels: np.ndarray, probabilities: np.ndarray | None = None
    ) -> None:
        """Append batch-level predictions and labels."""
        self.y_pred.extend(predictions.astype(int).tolist())
        self.y_true.extend(labels.astype(int).tolist())
        if probabilities is not None:
            self.y_prob.append(np.asarray(probabilities))

    def compute(self) -> dict:
        """Compute all configured metrics from accumulated state."""
        y_true = np.asarray(self.y_true)
        y_pred = np.asarray(self.y_pred)
        y_prob = np.concatenate(self.y_prob, axis=0) if self.y_prob else None

        per_class = {
            "sensitivity": {},
            "specificity": {},
            "auc_roc": {},
            "auc_pr": {},
            "f1": {},
        }

        cm = confusion_matrix(y_true, y_pred, labels=list(range(self.num_classes)))
        for class_idx, class_name in enumerate(self.class_names):
            tp = cm[class_idx, class_idx]
            fn = cm[class_idx, :].sum() - tp
            fp = cm[:, class_idx].sum() - tp
            tn = cm.sum() - (tp + fn + fp)
            sensitivity = tp / (tp + fn + 1e-12)
            specificity = tn / (tn + fp + 1e-12)
            per_class["sensitivity"][class_name] = float(sensitivity)
            per_class["specificity"][class_name] = float(specificity)
            per_class["f1"][class_name] = float(
                f1_score((y_true == class_idx).astype(int), (y_pred == class_idx).astype(int))
            )

            if y_prob is not None:
                y_true_bin = (y_true == class_idx).astype(int)
                try:
                    per_class["auc_roc"][class_name] = float(
                        roc_auc_score(y_true_bin, y_prob[:, class_idx])
                    )
                except ValueError:
                    per_class["auc_roc"][class_name] = float("nan")
                precision, recall, _ = precision_recall_curve(y_true_bin, y_prob[:, class_idx])
                per_class["auc_pr"][class_name] = float(sklearn_auc(recall, precision))

        return {
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
            "cohens_kappa": float(cohen_kappa_score(y_true, y_pred)),
            "per_class": per_class,
            "confusion_matrix": cm,
            "classification_report": classification_report(
                y_true, y_pred, target_names=list(self.class_names), output_dict=True
            ),
        }
