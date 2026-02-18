"""Tests for HAM10000Dataset behavior."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.dataset import HAM10000Dataset
from src.data.transforms import get_val_transforms


def test_dataset_returns_expected_sample_structure(tmp_path: Path) -> None:
    """Dataset returns dict payload with expected keys and shapes."""
    image_path = tmp_path / "ISIC_0001.jpg"
    Image.new("RGB", (64, 64), color=(100, 120, 140)).save(image_path)

    df = pd.DataFrame(
        {
            "image_id": ["ISIC_0001"],
            "lesion_id": ["lesion_a"],
            "dx": ["mel"],
            "age_norm": [0.45],
            "sex_idx": [0],
            "localization_idx": [1],
            "age_missing": [0],
            "sex_missing": [0],
            "localization_missing": [0],
        }
    )

    dataset = HAM10000Dataset(
        df=df,
        image_dir=tmp_path,
        transform=get_val_transforms(32),
        label_encoder={"mel": 0, "nv": 1},
    )
    sample = dataset[0]

    assert set(sample.keys()) == {"image", "metadata", "label", "image_id", "lesion_id"}
    assert sample["image"].shape == (3, 32, 32)
    assert sample["metadata"].ndim == 1
    assert isinstance(sample["label"], int)
    assert sample["image_id"] == "ISIC_0001"
    assert sample["lesion_id"] == "lesion_a"


def test_dataset_len_matches_dataframe_length(tmp_path: Path) -> None:
    """Dataset length equals dataframe row count."""
    for idx in range(2):
        Image.new("RGB", (32, 32), color=(idx, idx, idx)).save(tmp_path / f"ISIC_{idx}.jpg")

    df = pd.DataFrame(
        {
            "image_id": ["ISIC_0", "ISIC_1"],
            "lesion_id": ["l0", "l1"],
            "label": [0, 1],
            "age_norm": [0.2, 0.3],
            "sex_idx": [0, 1],
            "localization_idx": [0, 1],
            "age_missing": [0, 0],
            "sex_missing": [0, 0],
            "localization_missing": [0, 0],
        }
    )
    dataset = HAM10000Dataset(df=df, image_dir=tmp_path, transform=None)
    assert len(dataset) == 2
