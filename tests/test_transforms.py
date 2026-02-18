"""Tests for augmentation and deterministic validation transforms."""

from __future__ import annotations

import numpy as np

from src.data.transforms import get_train_transforms, get_tta_transforms, get_val_transforms


def test_train_transforms_produce_tensor_output() -> None:
    """Training transform pipeline should produce a CHW tensor."""
    image = np.random.randint(0, 256, size=(128, 128, 3), dtype=np.uint8)
    output = get_train_transforms(64)(image=image)["image"]
    assert tuple(output.shape) == (3, 64, 64)
    assert np.isfinite(output.numpy()).all()


def test_val_transforms_are_resize_normalize_only() -> None:
    """Validation transforms should remain deterministic and minimal."""
    transform = get_val_transforms(80)
    names = [t.__class__.__name__ for t in transform.transforms]
    assert names == ["Resize", "Normalize", "ToTensorV2"]


def test_tta_variants_count() -> None:
    """TTA list should include original plus four deterministic variants."""
    assert len(get_tta_transforms(64)) == 5
