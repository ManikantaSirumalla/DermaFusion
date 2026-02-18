"""Albumentations transform builders for dermatoscopy data."""

from __future__ import annotations

import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.data.preprocessing import shades_of_gray

__all__ = ["get_train_transforms", "get_tta_transforms", "get_val_transforms"]


def get_train_transforms(
    image_size: int = 380,
    use_elastic: bool = True,
    aspect_ratio: tuple[float, float] = (0.9, 1.1),
    apply_color_constancy: bool = True,
) -> A.Compose:
    """Return training-time dermoscopy augmentations (guide-aligned).

    When loading from preprocessed data (already color-corrected), set
    apply_color_constancy=False to avoid applying Shades of Gray again.
    """
    transforms_list: list = []
    if apply_color_constancy:
        transforms_list.append(A.Lambda(image=shades_of_gray))
    transforms_list.extend([
        A.RandomResizedCrop(
            image_size,
            image_size,
            scale=(0.8, 1.0),
            ratio=aspect_ratio,
        ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=15,
                border_mode=0,
                p=0.5,
            ),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.1,
                hue=0.05,
                p=0.5,
            ),
        ])
    if use_elastic:
        transforms_list.append(
            A.ElasticTransform(alpha=50, sigma=5, alpha_affine=0, p=0.3)
        )
    transforms_list.extend([
        A.OneOf(
            [
                A.GaussNoise(var_limit=(10, 50)),
                A.GaussianBlur(blur_limit=(3, 5)),
            ],
            p=0.3,
        ),
        A.CoarseDropout(max_holes=8, max_height=32, max_width=32, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return A.Compose(transforms_list)


def get_val_transforms(
    image_size: int = 380,
    apply_color_constancy: bool = True,
) -> A.Compose:
    """Return deterministic validation/test transforms (resize + normalize only)."""
    steps = []
    if apply_color_constancy:
        steps.append(A.Lambda(image=shades_of_gray))
    steps.extend([
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    return A.Compose(steps)


def get_tta_transforms(image_size: int = 380) -> list[A.Compose]:
    """Return 8x TTA variants (guide: original + 4 rotations + 2 flips + 1 flip+rot)."""
    normalize_to_tensor = [
        A.Resize(image_size, image_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
    base = A.Compose(normalize_to_tensor)
    return [
        base,
        A.Compose([A.HorizontalFlip(p=1.0), *normalize_to_tensor]),
        A.Compose([A.VerticalFlip(p=1.0), *normalize_to_tensor]),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), *normalize_to_tensor]),
        A.Compose([A.Rotate(limit=(180, 180), p=1.0), *normalize_to_tensor]),
        A.Compose([A.Rotate(limit=(270, 270), p=1.0), *normalize_to_tensor]),
        A.Compose([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), *normalize_to_tensor]),
        A.Compose([A.HorizontalFlip(p=1.0), A.Rotate(limit=(90, 90), p=1.0), *normalize_to_tensor]),
    ]
