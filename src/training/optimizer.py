"""Optimizer builders for multi-modal models."""

from __future__ import annotations

from typing import Any

import torch

__all__ = ["build_optimizer"]


def _cfg_value(cfg: Any, path: str, default: Any) -> Any:
    """Get nested value from dict/dataclass/omegaconf-like object."""
    current = cfg
    for key in path.split("."):
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def build_optimizer(model: torch.nn.Module, config: Any) -> torch.optim.Optimizer:
    """Create AdamW with lower LR for backbone and higher LR for heads."""
    base_lr = float(_cfg_value(config, "training.training.optimizer.lr", 1e-4))
    head_lr = float(_cfg_value(config, "training.training.optimizer.head_lr", 1e-3))
    weight_decay = float(_cfg_value(config, "training.training.optimizer.weight_decay", 0.01))

    backbone_params: list[torch.nn.Parameter] = []
    head_params: list[torch.nn.Parameter] = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "image_encoder.backbone" in name:
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": base_lr})
    if head_params:
        param_groups.append({"params": head_params, "lr": head_lr})

    return torch.optim.AdamW(param_groups, lr=head_lr, weight_decay=weight_decay)
