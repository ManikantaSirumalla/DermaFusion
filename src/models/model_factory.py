"""Factory utilities to construct full model variants from config."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from src.models.classifier import ClassificationHead
from src.models.fusion import CrossAttentionFusion, FiLMFusion, LateFusion
from src.models.image_encoder import ImageEncoder
from src.models.metadata_encoder import MetadataEncoder

__all__ = ["ImageOnlyModel", "MultiModalSkinCancerModel", "build_model"]


def _cfg_value(cfg: Any, path: str, default: Any) -> Any:
    """Get nested attribute/dict value with fallback."""
    current = cfg
    for key in path.split("."):
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


def _model_cfg_value(cfg: Any, key: str, default: Any) -> Any:
    """Read model field from either flat or nested Hydra model layout."""
    flat_value = _cfg_value(cfg, f"model.{key}", None)
    if flat_value is not None:
        return flat_value
    nested_value = _cfg_value(cfg, f"model.model.{key}", None)
    if nested_value is not None:
        return nested_value
    return default


class MultiModalSkinCancerModel(nn.Module):
    """Full multi-modal model combining image, metadata, fusion, and classifier."""

    def __init__(self, config: Any) -> None:
        """Initialize all model components from config."""
        super().__init__()
        backbone = _model_cfg_value(config, "backbone", "efficientnet_b4")
        pretrained = bool(_model_cfg_value(config, "pretrained", True))
        fusion_type = str(_model_cfg_value(config, "fusion", "late"))
        num_classes = int(_model_cfg_value(config, "num_classes", 7))

        self.image_encoder = ImageEncoder(backbone_name=backbone, pretrained=pretrained)
        self.metadata_encoder = MetadataEncoder(
            num_sex_categories=int(_model_cfg_value(config, "num_sex_categories", 3)),
            num_loc_categories=int(_model_cfg_value(config, "num_loc_categories", 32)),
            embedding_dim=int(_model_cfg_value(config, "metadata_embedding_dim", 32)),
            output_dim=int(_model_cfg_value(config, "metadata_output_dim", 128)),
        )

        image_dim = self.image_encoder.output_dim
        metadata_dim = self.metadata_encoder.output_dim
        if fusion_type == "late":
            self.fusion = LateFusion(image_dim=image_dim, metadata_dim=metadata_dim)
        elif fusion_type == "film":
            self.fusion = FiLMFusion(image_dim=image_dim, metadata_dim=metadata_dim)
        elif fusion_type == "cross_attention":
            self.fusion = CrossAttentionFusion(
                image_dim=image_dim,
                metadata_dim=metadata_dim,
                num_heads=int(_model_cfg_value(config, "cross_attention_heads", 4)),
            )
        else:
            raise ValueError(f"Unsupported fusion type: {fusion_type}")

        self.classifier = ClassificationHead(
            input_dim=self.fusion.output_dim,
            num_classes=num_classes,
            hidden_dim=int(_model_cfg_value(config, "classifier_hidden_dim", 512)),
            dropout=float(_model_cfg_value(config, "classifier_dropout", 0.4)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning class logits."""
        image_features = self.image_encoder(batch["image"])
        metadata_features = self.metadata_encoder(batch["metadata"])
        fused_features = self.fusion(image_features, metadata_features)
        return self.classifier(fused_features)


class ImageOnlyModel(nn.Module):
    """Image-only baseline model without metadata branch."""

    def __init__(self, config: Any) -> None:
        """Initialize image-only encoder and classifier."""
        super().__init__()
        backbone = _model_cfg_value(config, "backbone", "efficientnet_b4")
        pretrained = bool(_model_cfg_value(config, "pretrained", True))
        num_classes = int(_model_cfg_value(config, "num_classes", 7))
        self.image_encoder = ImageEncoder(backbone_name=backbone, pretrained=pretrained)
        self.classifier = ClassificationHead(
            input_dim=self.image_encoder.output_dim,
            num_classes=num_classes,
            hidden_dim=int(_model_cfg_value(config, "classifier_hidden_dim", 512)),
            dropout=float(_model_cfg_value(config, "classifier_dropout", 0.4)),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass returning class logits."""
        image_features = self.image_encoder(batch["image"])
        return self.classifier(image_features)


def build_model(config: Any) -> nn.Module:
    """Factory to build image-only or multimodal model from config."""
    use_metadata = bool(_model_cfg_value(config, "use_metadata", False))
    if use_metadata:
        return MultiModalSkinCancerModel(config)
    return ImageOnlyModel(config)
