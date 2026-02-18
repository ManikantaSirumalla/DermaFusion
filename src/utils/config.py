"""Hydra config loading and validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

__all__ = [
    "DataConfig",
    "ExperimentConfig",
    "ModelConfig",
    "TrainingConfig",
    "compose_config",
    "to_typed_config",
    "validate_config",
]


@dataclass(frozen=True)
class DataConfig:
    """Typed data configuration."""

    dataset_name: str
    image_size: int
    num_workers: int
    pin_memory: bool


@dataclass(frozen=True)
class ModelConfig:
    """Typed model configuration."""

    backbone: str
    pretrained: bool
    use_metadata: bool
    fusion: str
    num_classes: int


@dataclass(frozen=True)
class TrainingConfig:
    """Typed training configuration."""

    batch_size: int
    epochs: int
    mixed_precision: bool
    gradient_clip_val: float


@dataclass(frozen=True)
class ExperimentConfig:
    """Typed top-level experiment configuration."""

    seed: int
    experiment_name: str
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig


def compose_config(
    config_dir: Path,
    config_name: str = "config",
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load a Hydra config by composing defaults and optional overrides.

    Args:
        config_dir: Directory containing Hydra YAML configs.
        config_name: Root config name.
        overrides: Optional Hydra-style override list.

    Returns:
        Composed Hydra DictConfig.
    """
    with initialize_config_dir(config_dir=str(config_dir.resolve()), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides or [])
    return cfg


def validate_config(cfg: DictConfig) -> None:
    """Validate key configuration constraints and fail fast on invalid values.

    Args:
        cfg: Hydra/OmegaConf config object.

    Raises:
        ValueError: If required config values are missing or invalid.
    """
    required_top_keys = ("data", "model", "training", "seed", "experiment_name")
    for key in required_top_keys:
        if key not in cfg:
            raise ValueError(f"Missing required key: {key}")

    def _select(paths: list[str], default: Any) -> Any:
        for path in paths:
            value = OmegaConf.select(cfg, path, default=None)
            if value is not None:
                return value
        return default

    batch_size = int(_select(["training.batch_size", "training.training.batch_size"], 0))
    epochs = int(_select(["training.epochs", "training.training.epochs"], 0))
    image_size = int(_select(["data.image_size", "data.data.image_size"], 0))
    num_classes = int(_select(["model.num_classes", "model.model.num_classes"], -1))
    backbone = str(_select(["model.backbone", "model.model.backbone"], ""))
    use_metadata = bool(_select(["model.use_metadata", "model.model.use_metadata"], False))
    fusion = str(_select(["model.fusion", "model.model.fusion"], ""))

    if batch_size <= 0:
        raise ValueError("training.batch_size must be > 0")
    if epochs <= 0:
        raise ValueError("training.epochs must be > 0")
    if image_size <= 0:
        raise ValueError("data.image_size must be > 0")
    if num_classes != 7:
        raise ValueError("model.num_classes must be 7 for HAM10000 class set")

    allowed_backbones = {"efficientnet_b4", "swin_tiny", "convnext_v2"}
    if backbone not in allowed_backbones:
        raise ValueError(f"Unsupported backbone: {backbone}")

    allowed_fusions = {"late", "film", "cross_attention"}
    if use_metadata and fusion not in allowed_fusions:
        raise ValueError(f"Unsupported fusion type: {fusion}")


def to_typed_config(cfg: DictConfig) -> ExperimentConfig:
    """Convert DictConfig into typed dataclass representation.

    Args:
        cfg: Validated DictConfig object.

    Returns:
        Immutable typed experiment config.
    """
    cfg_dict: dict[str, Any] = OmegaConf.to_container(cfg, resolve=True)  # type: ignore[assignment]
    return ExperimentConfig(
        seed=int(cfg_dict["seed"]),
        experiment_name=str(cfg_dict["experiment_name"]),
        data=DataConfig(
            dataset_name=str(cfg_dict["data"]["dataset_name"]),
            image_size=int(cfg_dict["data"]["image_size"]),
            num_workers=int(cfg_dict["data"]["num_workers"]),
            pin_memory=bool(cfg_dict["data"]["pin_memory"]),
        ),
        model=ModelConfig(
            backbone=str(cfg_dict["model"]["backbone"]),
            pretrained=bool(cfg_dict["model"]["pretrained"]),
            use_metadata=bool(cfg_dict["model"]["use_metadata"]),
            fusion=str(cfg_dict["model"]["fusion"]),
            num_classes=int(cfg_dict["model"]["num_classes"]),
        ),
        training=TrainingConfig(
            batch_size=int(cfg_dict["training"]["batch_size"]),
            epochs=int(cfg_dict["training"]["epochs"]),
            mixed_precision=bool(cfg_dict["training"]["mixed_precision"]),
            gradient_clip_val=float(cfg_dict["training"]["gradient_clip_val"]),
        ),
    )

