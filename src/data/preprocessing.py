"""Metadata preprocessing, color constancy, and leakage-safe split helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

__all__ = [
    "MetadataEncodingStats",
    "SplitResult",
    "compute_class_weights",
    "create_splits",
    "encode_metadata",
    "load_metadata",
    "shades_of_gray",
    "dull_razor",
]


def dull_razor(image: np.ndarray, **_: object) -> np.ndarray:
    """DullRazor-style hair removal: detect dark thin structures and inpaint.

    Uses grayscale, morphological black-hat with linear SE to highlight hair,
    threshold to mask, then OpenCV inpainting to fill hair regions.
    """
    try:
        import cv2
    except ImportError:
        return image

    if image.ndim == 2:
        gray = image
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Linear structuring element (hair is thin and long); length ~15–25 px
    kernel_len = 21
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, kernel_len))
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    # Optional: also use a short horizontal kernel for cross-hairs
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_len, 1))
    blackhat_h = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel_h)
    blackhat = np.maximum(blackhat, blackhat_h)

    # Threshold: dark thin structures (hair) have high black-hat response
    _, mask = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    mask = np.uint8(mask)
    # Dilate slightly so inpainting covers full hair width
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

    if mask.sum() == 0:
        return image

    # Inpaint: fill mask regions using surrounding pixels (Telea or NS)
    if image.ndim == 2:
        out = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    else:
        out = cv2.inpaint(image, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    return out


@dataclass(frozen=True)
class SplitResult:
    """Container for train/val folds and fixed test indices."""

    train_val_folds: list[tuple[np.ndarray, np.ndarray]]
    test_indices: np.ndarray


@dataclass(frozen=True)
class MetadataEncodingStats:
    """Reusable statistics and mappings for deterministic metadata encoding."""

    age_median: float
    sex_mapping: dict[str, int]
    localization_mapping: dict[str, int]


def shades_of_gray(image: np.ndarray, power: int = 6, **_: object) -> np.ndarray:
    """Apply Shades of Gray color constancy normalization.

    Args:
        image: Input RGB image array in uint8 format.
        power: Exponent used by Shades of Gray algorithm.

    Returns:
        Color-normalized uint8 RGB image.
    """
    img = image.astype(np.float32)
    norm = np.power(img, power)
    mean_per_channel = norm.mean(axis=(0, 1))
    mean_per_channel = np.power(mean_per_channel, 1.0 / power)
    gray = np.mean(mean_per_channel)
    scale = gray / (mean_per_channel + 1e-7)
    result = img * scale[np.newaxis, np.newaxis, :]
    return np.clip(result, 0, 255).astype(np.uint8)


def load_metadata(csv_path: str | Path) -> pd.DataFrame:
    """Load metadata and validate required columns.

    Args:
        csv_path: Path to metadata CSV.

    Returns:
        Metadata DataFrame.

    Raises:
        ValueError: If required columns are missing.
    """
    required = {"lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization"}
    df = pd.read_csv(csv_path)
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required metadata columns: {sorted(missing)}")
    return df.copy()


def _build_stats(df: pd.DataFrame) -> MetadataEncodingStats:
    """Build encoding statistics from a training split only."""
    age_series = pd.to_numeric(df["age"], errors="coerce")
    age_median = float(age_series.median()) if not age_series.dropna().empty else 0.0

    sex_mapping = {"male": 0, "female": 1, "unknown": 2}

    loc_values = (
        df["localization"].fillna("unknown").astype(str).str.strip().str.lower().tolist()
    )
    unique_localizations = sorted(set(loc_values))
    if "unknown" not in unique_localizations:
        unique_localizations.insert(0, "unknown")
    localization_mapping = {name: idx for idx, name in enumerate(unique_localizations)}

    return MetadataEncodingStats(
        age_median=age_median,
        sex_mapping=sex_mapping,
        localization_mapping=localization_mapping,
    )


def encode_metadata(
    df: pd.DataFrame,
    stats: MetadataEncodingStats | None = None,
) -> tuple[pd.DataFrame, MetadataEncodingStats]:
    """Encode metadata while preserving missingness indicators.

    If `stats` is None, statistics are computed from `df` (training mode).
    For validation/test, pass training-derived `stats` to avoid leakage.

    Args:
        df: Raw metadata DataFrame.
        stats: Optional precomputed training statistics.

    Returns:
        Tuple of encoded DataFrame and encoding stats used.
    """
    fitted_stats = stats or _build_stats(df)

    out = df.copy()
    out["age"] = pd.to_numeric(out["age"], errors="coerce")
    out["age_missing"] = out["age"].isna().astype(int)
    out["age"] = out["age"].fillna(fitted_stats.age_median)
    out["age_norm"] = (out["age"].clip(lower=0.0, upper=100.0) / 100.0).astype(np.float32)

    out["sex_missing"] = out["sex"].isna().astype(int)
    out["sex"] = out["sex"].fillna("unknown").astype(str).str.strip().str.lower()
    out["sex"] = out["sex"].where(out["sex"].isin(fitted_stats.sex_mapping), "unknown")
    out["sex_idx"] = out["sex"].map(fitted_stats.sex_mapping).astype(int)

    out["localization_missing"] = out["localization"].isna().astype(int)
    out["localization"] = (
        out["localization"].fillna("unknown").astype(str).str.strip().str.lower()
    )
    out["localization_idx"] = out["localization"].map(fitted_stats.localization_mapping)
    unknown_loc_idx = fitted_stats.localization_mapping.get("unknown", 0)
    out["localization_idx"] = out["localization_idx"].fillna(unknown_loc_idx).astype(int)

    return out, fitted_stats


def compute_class_weights(labels: Iterable[int]) -> torch.Tensor:
    """Compute normalized inverse-frequency class weights."""
    label_array = np.array(list(labels), dtype=np.int64)
    classes, counts = np.unique(label_array, return_counts=True)
    weights = np.zeros(int(classes.max()) + 1, dtype=np.float32)
    inv = 1.0 / counts.astype(np.float32)
    inv = inv / inv.sum() * len(classes)
    weights[classes] = inv
    return torch.tensor(weights, dtype=torch.float32)


def create_splits(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 42,
    test_size: float = 0.15,
) -> SplitResult:
    """Create lesion-group-safe train/val folds with fixed test holdout."""
    groups = df["lesion_id"].astype(str).to_numpy()
    indices = np.arange(len(df))

    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_val_idx, test_idx = next(gss.split(indices, groups=groups))
    train_val_df = df.iloc[train_val_idx].reset_index(drop=False)

    gkf = GroupKFold(n_splits=n_folds)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    tv_groups = train_val_df["lesion_id"].astype(str).to_numpy()
    tv_indices = np.arange(len(train_val_df))

    for train_local, val_local in gkf.split(tv_indices, groups=tv_groups):
        train_indices = train_val_df.iloc[train_local]["index"].to_numpy()
        val_indices = train_val_df.iloc[val_local]["index"].to_numpy()
        folds.append((train_indices, val_indices))

        train_ids = set(df.iloc[train_indices]["lesion_id"])
        val_ids = set(df.iloc[val_indices]["lesion_id"])
        test_ids = set(df.iloc[test_idx]["lesion_id"])
        assert train_ids.isdisjoint(val_ids), "DATA LEAKAGE: train/val overlap!"
        assert train_ids.isdisjoint(test_ids), "DATA LEAKAGE: train/test overlap!"
        assert val_ids.isdisjoint(test_ids), "DATA LEAKAGE: val/test overlap!"

    return SplitResult(train_val_folds=folds, test_indices=test_idx)
