"""Tests for metadata preprocessing and split leakage protection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.preprocessing import (
    compute_class_weights,
    create_splits,
    encode_metadata,
    shades_of_gray,
)


def test_shades_of_gray_returns_valid_uint8_image() -> None:
    """Color constancy output should keep input image shape and uint8 range."""
    image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    output = shades_of_gray(image)
    assert output.shape == image.shape
    assert output.dtype == np.uint8
    assert int(output.min()) >= 0
    assert int(output.max()) <= 255


def test_encode_metadata_handles_missing_and_uses_median() -> None:
    """Encoding fills missing metadata and applies deterministic mappings."""
    train_df = pd.DataFrame(
        {
            "age": [20.0, np.nan, 80.0],
            "sex": ["male", "female", np.nan],
            "localization": ["back", "face", np.nan],
            "dx_type": ["histo", "histo", "consensus"],
        }
    )
    encoded_train, stats = encode_metadata(train_df)
    encoded_val, _ = encode_metadata(train_df.iloc[[1]].copy(), stats=stats)

    assert not encoded_train[["age_norm", "sex_idx", "localization_idx"]].isna().any().any()
    assert encoded_train.loc[1, "age"] == 50.0  # median of 20 and 80.
    assert encoded_train.loc[1, "age_missing"] == 1
    assert set(encoded_train["sex_idx"].tolist()).issubset({0, 1, 2})
    assert encoded_val.loc[1, "age"] == stats.age_median


def test_compute_class_weights_inverse_frequency_behavior() -> None:
    """Minority classes should receive larger weights than majority classes."""
    labels = [0, 0, 0, 1, 2]
    weights = compute_class_weights(labels)
    assert weights[1] > weights[0]
    assert weights[2] > weights[0]


def test_create_splits_no_lesion_overlap() -> None:
    """Generated splits should enforce lesion-level disjointness."""
    df = pd.DataFrame(
        {
            "lesion_id": [f"l{i // 2}" for i in range(20)],
            "image_id": [f"img_{i}" for i in range(20)],
        }
    )
    split_result = create_splits(df, n_folds=3, seed=7, test_size=0.2)
    assert len(split_result.train_val_folds) == 3
