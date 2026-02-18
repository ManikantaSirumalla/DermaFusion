"""Tests for class-balanced lesion-aware sampler."""

from __future__ import annotations

from collections import Counter

import pandas as pd

from src.data.sampler import ClassBalancedSampler


def test_sampler_balances_class_counts_approximately() -> None:
    """Sampler output should flatten class imbalance."""
    df = pd.DataFrame(
        {
            "label": [0, 0, 0, 0, 1, 1, 2, 2],
            "lesion_id": ["l0", "l1", "l2", "l3", "l4", "l5", "l6", "l7"],
        }
    )
    sampler = ClassBalancedSampler(df, batch_size=4, seed=123)
    indices = list(iter(sampler))
    sampled_labels = [int(df.iloc[i]["label"]) for i in indices]
    counts = Counter(sampled_labels)
    assert max(counts.values()) - min(counts.values()) <= 2


def test_sampler_avoids_same_lesion_in_batch_when_possible() -> None:
    """Sampler should avoid duplicate lesion IDs within each mini-batch."""
    df = pd.DataFrame(
        {
            "label": [0, 0, 1, 1, 2, 2],
            "lesion_id": ["l0", "l1", "l2", "l3", "l4", "l5"],
        }
    )
    sampler = ClassBalancedSampler(df, batch_size=3, seed=42)
    indices = list(iter(sampler))
    for i in range(0, len(indices), 3):
        batch = indices[i : i + 3]
        lesions = [str(df.iloc[idx]["lesion_id"]) for idx in batch]
        assert len(lesions) == len(set(lesions))
