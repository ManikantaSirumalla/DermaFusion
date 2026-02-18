"""Evaluation package exports."""

from src.evaluation.calibration import (
    expected_calibration_error,
    plot_reliability_diagram,
    temperature_scaling,
)
from src.evaluation.fairness import demographic_parity, equalized_odds, per_group_performance
from src.evaluation.interpretability import (
    attention_rollout,
    generate_gradcam,
    generate_interpretation_report,
    visualize_gradcam,
)
from src.evaluation.metrics import MetricCalculator
from src.evaluation.statistical import bootstrap_confidence_intervals, compute_all_cis, mcnemar_test

__all__ = [
    "MetricCalculator",
    "attention_rollout",
    "bootstrap_confidence_intervals",
    "compute_all_cis",
    "demographic_parity",
    "equalized_odds",
    "expected_calibration_error",
    "generate_gradcam",
    "generate_interpretation_report",
    "mcnemar_test",
    "per_group_performance",
    "plot_reliability_diagram",
    "temperature_scaling",
    "visualize_gradcam",
]
