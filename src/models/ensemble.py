"""Model ensembling strategies (guide: E1 single vs 3-model ensemble)."""

from __future__ import annotations

import torch
from torch import nn

__all__ = ["ensemble_predict"]


def ensemble_predict(
    models: list[nn.Module],
    batch: dict,
    device: torch.device,
    average: str = "softmax",
) -> torch.Tensor:
    """Average logits or probabilities from multiple models (guide: E1).

    Args:
        models: List of models (e.g. EfficientNet, Swin, ConvNeXt image-only).
        batch: Input batch dict with 'image' and optionally 'metadata'.
        device: Device to run on.
        average: 'softmax' to average after softmax (recommended), or 'logits'.

    Returns:
        Averaged logits [batch_size, num_classes].
    """
    logits_list: list[torch.Tensor] = []
    for model in models:
        model.eval()
        with torch.no_grad():
            out = model(batch)
            logits_list.append(out.to(device))
    stacked = torch.stack(logits_list, dim=0)
    if average == "softmax":
        probs = torch.softmax(stacked, dim=-1)
        avg_probs = probs.mean(dim=0)
        return torch.log(avg_probs.clamp(min=1e-12))
    return stacked.mean(dim=0)
