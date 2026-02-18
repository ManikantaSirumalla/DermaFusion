"""Dataset module for multi-modal HAM10000/ISIC samples."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

__all__ = ["HAM10000Dataset"]


class HAM10000Dataset(Dataset):
    """Multi-modal dataset for HAM10000-style metadata + dermoscopic images.

    Returns:
        Dict with keys:
            - image: torch.Tensor [3, H, W]
            - metadata: torch.Tensor [num_meta_features]
            - label: int
            - image_id: str
            - lesion_id: str
    """

    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str | Path,
        transform: Any | None,
        metadata_columns: Sequence[str] | None = None,
        label_encoder: Mapping[str, int] | None = None,
        validate_images_on_init: bool = True,
    ) -> None:
        """Initialize dataset.

        Args:
            df: Preprocessed metadata DataFrame.
            image_dir: Root directory containing image files.
            transform: Albumentations transform or None.
            metadata_columns: Ordered metadata columns for output tensor.
            label_encoder: Optional mapping from diagnosis string to class index.
            validate_images_on_init: If True, open and verify every image at init
                (slow for large datasets). If False, skip; invalid images may
                surface at load time.
        """
        self.df = df.reset_index(drop=True).copy()
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.label_encoder = dict(label_encoder) if label_encoder is not None else None
        self.metadata_columns = list(
            metadata_columns
            if metadata_columns is not None
            else [
                "age_norm",
                "sex_idx",
                "localization_idx",
                "age_missing",
                "sex_missing",
                "localization_missing",
            ]
        )
        missing_columns = [c for c in self.metadata_columns if c not in self.df.columns]
        if missing_columns:
            raise ValueError(f"Missing required metadata columns: {missing_columns}")
        if not self.image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {self.image_dir}")
        if validate_images_on_init:
            self._filter_invalid_samples()

    def _filter_invalid_samples(self) -> None:
        """Remove missing/corrupted images and log what was skipped."""
        logger = logging.getLogger(__name__)
        keep_indices: list[int] = []
        skipped: list[str] = []

        for idx in range(len(self.df)):
            row = self.df.iloc[idx]
            image_id = str(row["image_id"])
            subdir = row.get("image_subdir")
            subdir = None if pd.isna(subdir) else str(subdir)
            try:
                image_path = self._resolve_image_path(image_id, subdir=subdir)
                with Image.open(image_path) as image:
                    image.verify()
                keep_indices.append(idx)
            except (FileNotFoundError, UnidentifiedImageError, OSError):
                skipped.append(image_id)

        if skipped:
            self.df = self.df.iloc[keep_indices].reset_index(drop=True)
            preview = ", ".join(skipped[:5])
            logger.warning(
                "Skipped %d invalid images from %s. First examples: %s",
                len(skipped),
                self.image_dir,
                preview,
            )

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.df)

    def _resolve_image_path(self, image_id: str, subdir: str | None = None) -> Path:
        """Resolve image file path from image_id; if subdir is set, look under image_dir/subdir."""
        base = (self.image_dir / subdir) if subdir else self.image_dir
        base = Path(base)
        image_path = base / image_id
        if image_path.exists():
            return image_path
        jpg_path = base / f"{image_id}.jpg"
        if jpg_path.exists():
            return jpg_path
        raise FileNotFoundError(f"Image not found for image_id='{image_id}' in {base}")

    def _get_label(self, row: pd.Series) -> int:
        """Extract encoded label from row."""
        if "label" in row:
            return int(row["label"])
        if "dx" in row:
            dx = str(row["dx"])
            if self.label_encoder is None:
                raise ValueError("label_encoder is required when labels are provided via 'dx'.")
            if dx not in self.label_encoder:
                raise KeyError(f"Diagnosis '{dx}' not found in label_encoder.")
            return int(self.label_encoder[dx])
        raise KeyError("Expected either 'label' or 'dx' column in metadata DataFrame.")

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Get one sample."""
        row = self.df.iloc[idx]
        image_id = str(row["image_id"])
        lesion_id = str(row["lesion_id"])
        subdir = row.get("image_subdir")
        subdir = None if pd.isna(subdir) else str(subdir)
        image_path = self._resolve_image_path(image_id, subdir=subdir)

        with Image.open(image_path) as image:
            image_np = np.array(image.convert("RGB"))

        if self.transform is not None:
            transformed = self.transform(image=image_np)
            image_tensor = transformed["image"]
        else:
            image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0

        metadata_values = (
            row[self.metadata_columns].fillna(0.0).to_numpy(dtype=np.float32, copy=True)
        )
        metadata_tensor = torch.tensor(metadata_values, dtype=torch.float32)
        label = self._get_label(row)

        return {
            "image": image_tensor,
            "metadata": metadata_tensor,
            "label": label,
            "image_id": image_id,
            "lesion_id": lesion_id,
        }
