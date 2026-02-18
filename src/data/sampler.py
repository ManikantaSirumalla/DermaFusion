"""Sampling utilities for class balancing with lesion-group awareness."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterator

import numpy as np
import pandas as pd
from torch.utils.data import Sampler

__all__ = ["ClassBalancedSampler"]


class ClassBalancedSampler(Sampler[int]):
    """Class-balanced sampler that samples by lesion groups first.

    This sampler upweights minority classes by sampling lesion IDs to a balanced
    class target, then chooses one image index per sampled lesion.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        label_col: str = "label",
        lesion_col: str = "lesion_id",
        batch_size: int = 32,
        seed: int = 42,
    ) -> None:
        """Initialize the class-balanced lesion-aware sampler."""
        self.df = df.reset_index(drop=True)
        self.label_col = label_col
        self.lesion_col = lesion_col
        self.batch_size = batch_size
        self.seed = seed

        if self.label_col not in self.df.columns:
            raise ValueError(f"Missing label column '{self.label_col}' in sampler dataframe.")
        if self.lesion_col not in self.df.columns:
            raise ValueError(f"Missing lesion column '{self.lesion_col}' in sampler dataframe.")

        self._lesion_to_indices: dict[str, list[int]] = defaultdict(list)
        self._lesion_to_label: dict[str, int] = {}
        for idx, row in self.df.iterrows():
            lesion_id = str(row[self.lesion_col])
            self._lesion_to_indices[lesion_id].append(int(idx))
            self._lesion_to_label[lesion_id] = int(row[self.label_col])

        self._class_to_lesions: dict[int, list[str]] = defaultdict(list)
        for lesion_id, label in self._lesion_to_label.items():
            self._class_to_lesions[label].append(lesion_id)

        self._classes = sorted(self._class_to_lesions.keys())
        self._target_per_class = max(len(v) for v in self._class_to_lesions.values())
        self._num_samples = self._target_per_class * len(self._classes)

    def __len__(self) -> int:
        """Return number of sampled indices per epoch."""
        return self._num_samples

    def _balance_lesion_list(self, rng: np.random.Generator) -> list[str]:
        """Sample lesion IDs per class with replacement to balance class counts."""
        lesion_order: list[str] = []
        for cls in self._classes:
            lesions = self._class_to_lesions[cls]
            sampled = rng.choice(lesions, size=self._target_per_class, replace=True)
            lesion_order.extend(sampled.tolist())
        rng.shuffle(lesion_order)
        return lesion_order

    def _avoid_in_batch_duplicates(self, lesion_order: list[str]) -> list[str]:
        """Reduce repeated lesion IDs inside the same batch when possible."""
        if self.batch_size <= 1:
            return lesion_order
        fixed: list[str] = []
        queue = lesion_order[:]
        while queue:
            batch: list[str] = []
            seen: set[str] = set()
            spillover: list[str] = []
            while queue and len(batch) < self.batch_size:
                lesion = queue.pop(0)
                if lesion in seen:
                    spillover.append(lesion)
                    continue
                batch.append(lesion)
                seen.add(lesion)
            fixed.extend(batch)
            queue = spillover + queue
        return fixed

    def __iter__(self) -> Iterator[int]:
        """Yield balanced image indices for one epoch."""
        rng = np.random.default_rng(self.seed)
        lesion_order = self._balance_lesion_list(rng)
        lesion_order = self._avoid_in_batch_duplicates(lesion_order)

        indices: list[int] = []
        for lesion_id in lesion_order:
            candidate_indices = self._lesion_to_indices[lesion_id]
            sampled_idx = int(rng.choice(candidate_indices))
            indices.append(sampled_idx)
        return iter(indices)
