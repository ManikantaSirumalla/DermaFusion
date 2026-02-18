"""Organize downloaded ISIC 2018 Task 3 files into project raw-data structure."""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logger

REQUIRED_METADATA_COLUMNS = {
    "lesion_id",
    "image_id",
    "dx",
    "dx_type",
    "age",
    "sex",
    "localization",
}

ISIC_REQUIRED_DIRS = (
    "ISIC2018_Task3_Training_Input",
    "ISIC2018_Task3_Validation_Input",
    "ISIC2018_Task3_Test_Input",
)

GROUND_TRUTH_LABEL_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for data setup."""
    parser = argparse.ArgumentParser(description="Setup ISIC 2018 Task 3 dataset structure.")
    parser.add_argument(
        "--raw-data-dir",
        type=Path,
        required=True,
        help="Path containing downloaded ISIC 2018 Task 3 files/folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw",
        help="Output path where data/raw/images and data/raw/metadata are created.",
    )
    parser.add_argument(
        "--materialize-mode",
        type=str,
        choices=["copy", "symlink"],
        default="copy",
        help="How to materialize image files into output images/* directories.",
    )
    parser.add_argument(
        "--ham-metadata-csv",
        type=Path,
        default=None,
        help=(
            "Optional path to HAM10000_metadata.csv for age/sex/localization/dx_type. "
            "If omitted, script will try to auto-discover it near datasets."
        ),
    )
    return parser.parse_args()


def _materialize_images(image_paths: list[Path], target_image_dir: Path, mode: str) -> None:
    """Copy or symlink images into a target image directory."""
    target_image_dir.mkdir(parents=True, exist_ok=True)
    for src_path in image_paths:
        dst_path = target_image_dir / src_path.name
        if dst_path.exists():
            continue
        if mode == "copy":
            shutil.copy2(src_path, dst_path)
        else:
            dst_path.symlink_to(src_path.resolve())


def _validate_required_items(raw_data_dir: Path) -> None:
    """Validate that all required ISIC Task 3 files/folders are present."""
    missing = [name for name in ISIC_REQUIRED_DIRS if not (raw_data_dir / name).exists()]
    if not (raw_data_dir / "ISIC2018_Task3_Training_LesionGroupings.csv").exists():
        missing.append("ISIC2018_Task3_Training_LesionGroupings.csv")
    if missing:
        raise FileNotFoundError(
            "Missing required ISIC 2018 Task 3 files/directories: "
            f"{sorted(missing)}"
        )


def _validate_ground_truth_columns(ground_truth_csv: Path) -> None:
    """Ensure ISIC ground truth CSV includes all 7 class columns."""
    gt_df = pd.read_csv(ground_truth_csv)
    missing = [c for c in GROUND_TRUTH_LABEL_COLS if c not in gt_df.columns]
    if missing:
        raise ValueError(
            f"Ground truth CSV missing expected diagnosis columns: {missing}"
        )


def _validate_lesion_groupings(groupings_csv: Path) -> None:
    """Ensure lesion groupings CSV exists and has lesion_id information."""
    groupings_df = pd.read_csv(groupings_csv)
    required_group_cols = {"image", "lesion_id"}
    missing = required_group_cols.difference(groupings_df.columns)
    if missing:
        raise ValueError(
            f"Lesion groupings CSV missing required columns: {sorted(missing)}"
        )


def _resolve_ground_truth_csv(raw_data_dir: Path, split: str) -> Path:
    """Resolve ground-truth CSV path for a given split.

    Supports both layouts:
    - `<root>/ISIC2018_Task3_<Split>_GroundTruth.csv`
    - `<root>/ISIC2018_Task3_<Split>_GroundTruth/ISIC2018_Task3_<Split>_GroundTruth.csv`
    """
    file_name = f"ISIC2018_Task3_{split}_GroundTruth.csv"
    flat_csv = raw_data_dir / file_name
    if flat_csv.exists():
        return flat_csv

    nested_csv = raw_data_dir / f"ISIC2018_Task3_{split}_GroundTruth" / file_name
    if nested_csv.exists():
        return nested_csv

    raise FileNotFoundError(f"Could not locate ground-truth CSV for split '{split}'.")


def _find_ham_metadata(raw_data_dir: Path, explicit_csv: Path | None) -> Path | None:
    """Find HAM10000 metadata CSV from explicit path or nearby dataset folders."""
    if explicit_csv is not None:
        return explicit_csv if explicit_csv.exists() else None

    local_candidate = raw_data_dir / "HAM10000_metadata.csv"
    if local_candidate.exists():
        return local_candidate

    datasets_root = raw_data_dir.parent
    for candidate in sorted(datasets_root.rglob("HAM10000_metadata.csv")):
        return candidate
    return None


def _verify_image_files(image_paths: list[Path], logger: logging.Logger) -> list[Path]:
    """Return list of unreadable/corrupted images after PIL validation."""
    bad_images: list[Path] = []
    for image_path in image_paths:
        try:
            with Image.open(image_path) as img:
                img.verify()
        except Exception:
            bad_images.append(image_path)
    if bad_images:
        logger.warning("Detected %d unreadable image files.", len(bad_images))
    return bad_images


def _merge_isic_metadata(
    raw_data_dir: Path,
    output_meta_dir: Path,
    train_gt_csv: Path,
    ham_metadata_csv: Path | None,
) -> pd.DataFrame:
    """Build canonical metadata table by merging available ISIC/HAM10000 metadata files."""
    groupings = pd.read_csv(raw_data_dir / "ISIC2018_Task3_Training_LesionGroupings.csv")
    gt = pd.read_csv(train_gt_csv)
    merged = gt.merge(groupings, on="image", how="left")

    if ham_metadata_csv is not None:
        ham_df = pd.read_csv(ham_metadata_csv)
        ham_df = ham_df.rename(columns={"image_id": "image"})
        merged = merged.merge(
            ham_df[["image", "dx", "dx_type", "age", "sex", "localization"]],
            on="image",
            how="left",
        )

    merged = merged.rename(columns={"image": "image_id"})
    missing_cols = REQUIRED_METADATA_COLUMNS.difference(merged.columns)
    if missing_cols:
        raise ValueError(
            "Merged metadata is missing required columns. "
            "Ensure HAM10000_metadata.csv is available via --ham-metadata-csv "
            "or under datasets/. "
            f"Missing: {sorted(missing_cols)}"
        )

    output_meta_dir.mkdir(parents=True, exist_ok=True)
    out_csv = output_meta_dir / "metadata_merged.csv"
    merged.to_csv(out_csv, index=False)
    return merged


def setup_data(
    raw_data_dir: Path,
    output_dir: Path,
    materialize_mode: str,
    ham_metadata_csv: Path | None,
    logger: logging.Logger,
) -> None:
    """Organize downloaded ISIC 2018 Task 3 files into canonical project layout."""
    _validate_required_items(raw_data_dir)

    training_input = raw_data_dir / "ISIC2018_Task3_Training_Input"
    validation_input = raw_data_dir / "ISIC2018_Task3_Validation_Input"
    test_input = raw_data_dir / "ISIC2018_Task3_Test_Input"

    train_gt = _resolve_ground_truth_csv(raw_data_dir, "Training")
    val_gt = _resolve_ground_truth_csv(raw_data_dir, "Validation")
    test_gt = _resolve_ground_truth_csv(raw_data_dir, "Test")
    lesion_groupings = raw_data_dir / "ISIC2018_Task3_Training_LesionGroupings.csv"
    discovered_ham_metadata = _find_ham_metadata(raw_data_dir, ham_metadata_csv)
    if discovered_ham_metadata is None:
        logger.warning(
            "HAM10000_metadata.csv not found automatically; metadata merge may fail for "
            "age/sex/localization columns."
        )
    else:
        logger.info("Using HAM metadata from: %s", discovered_ham_metadata)

    _validate_ground_truth_columns(train_gt)
    _validate_ground_truth_columns(val_gt)
    _validate_ground_truth_columns(test_gt)
    _validate_lesion_groupings(lesion_groupings)

    train_images = sorted(training_input.glob("*.jpg"))
    val_images = sorted(validation_input.glob("*.jpg"))
    test_images = sorted(test_input.glob("*.jpg"))
    if len(train_images) != 10015:
        raise ValueError(f"Expected 10015 training images, found {len(train_images)}")

    images_root = output_dir / "images"
    metadata_root = output_dir / "metadata"

    _materialize_images(train_images, images_root / "train", materialize_mode)
    _materialize_images(val_images, images_root / "val", materialize_mode)
    _materialize_images(test_images, images_root / "test", materialize_mode)

    metadata_root.mkdir(parents=True, exist_ok=True)
    for csv_file in [train_gt, val_gt, test_gt, lesion_groupings]:
        shutil.copy2(csv_file, metadata_root / csv_file.name)

    merged_df = _merge_isic_metadata(
        raw_data_dir=raw_data_dir,
        output_meta_dir=metadata_root,
        train_gt_csv=train_gt,
        ham_metadata_csv=discovered_ham_metadata,
    )

    bad_train = _verify_image_files(train_images, logger)
    bad_val = _verify_image_files(val_images, logger)
    bad_test = _verify_image_files(test_images, logger)
    if bad_train or bad_val or bad_test:
        logger.warning("Corrupted image files found and logged; please inspect source data.")

    logger.info(
        "ISIC data setup complete: train=%d, val=%d, test=%d, metadata_rows=%d",
        len(train_images),
        len(val_images),
        len(test_images),
        len(merged_df),
    )


def main() -> None:
    """Entrypoint for ISIC 2018 Task 3 data setup."""
    args = parse_args()
    logger = setup_logger(
        name="setup_data",
        level=logging.INFO,
        log_file=PROJECT_ROOT / "outputs" / "logs" / "setup_data.log",
    )

    if not args.raw_data_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {args.raw_data_dir}")

    setup_data(
        raw_data_dir=args.raw_data_dir,
        output_dir=args.output_dir,
        materialize_mode=args.materialize_mode,
        ham_metadata_csv=args.ham_metadata_csv,
        logger=logger,
    )


if __name__ == "__main__":
    main()
