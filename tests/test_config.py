"""Step 1 config loading and validation tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.utils.config import compose_config, to_typed_config, validate_config


def test_compose_and_validate_default_config() -> None:
    """Config composes from Hydra defaults and passes validation."""
    cfg = compose_config(config_dir=Path("configs"))
    validate_config(cfg)
    typed_cfg = to_typed_config(cfg)
    assert typed_cfg.model.num_classes == 7
    assert typed_cfg.training.batch_size > 0


def test_validate_rejects_invalid_num_classes() -> None:
    """Validation fails on invalid class count."""
    cfg = compose_config(config_dir=Path("configs"), overrides=["model.num_classes=6"])
    with pytest.raises(ValueError, match="model.num_classes must be 7"):
        validate_config(cfg)

