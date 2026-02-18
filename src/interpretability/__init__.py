# Interpretability module (guide: gradcam, attention, SHAP).
# Re-exports from evaluation.interpretability and adds SHAP stub.

from src.evaluation.interpretability import (
    attention_rollout,
    generate_gradcam,
    generate_interpretation_report,
    visualize_gradcam,
)

__all__ = [
    "attention_rollout",
    "generate_gradcam",
    "generate_interpretation_report",
    "visualize_gradcam",
    "shap_metadata_waterfall",
]


def shap_metadata_waterfall(
    model: object,
    metadata_tensor: object,
    feature_names: list[str],
    save_path: object = None,
) -> None:
    """SHAP waterfall for metadata branch (guide: interpretability deliverable).
    Stub: implement with shap.Explainer on metadata encoder output.
    """
    _ = model
    _ = metadata_tensor
    _ = feature_names
    _ = save_path
    raise NotImplementedError("SHAP metadata analysis: install shap and implement Explainer")
