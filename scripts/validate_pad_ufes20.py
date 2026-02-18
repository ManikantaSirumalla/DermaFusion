"""Validate PAD-UFES-20 dataset integrity and metadata consistency."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for PAD-UFES-20 validation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("datasets/PAD-UFES-20"),
        help="PAD-UFES-20 root directory.",
    )
    parser.add_argument(
        "--metadata-file",
        type=str,
        default="metadata.csv",
        help="Metadata CSV filename under dataset-dir.",
    )
    parser.add_argument(
        "--images-subdir",
        type=str,
        default="zr7vgbcyr2-1/images",
        help="Relative images folder containing PNG parts.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to save full validation report as JSON.",
    )
    return parser.parse_args()


def _load_metadata_rows(metadata_path: Path) -> list[dict[str, str]]:
    """Load metadata rows from CSV file."""
    with metadata_path.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return [row for row in reader]


def _build_report(dataset_dir: Path, metadata_path: Path, images_dir: Path) -> dict[str, Any]:
    """Compute dataset integrity report."""
    rows = _load_metadata_rows(metadata_path)
    image_files = sorted(images_dir.rglob("*.png"))

    metadata_img_ids = [row.get("img_id", "").strip() for row in rows if row.get("img_id")]
    metadata_img_id_set = set(metadata_img_ids)
    image_name_set = {path.name for path in image_files}

    duplicate_img_ids = sorted(
        img_id for img_id, count in Counter(metadata_img_ids).items() if count > 1
    )
    missing_images = sorted(metadata_img_id_set - image_name_set)
    orphan_images = sorted(image_name_set - metadata_img_id_set)

    diagnostic_counts = Counter(
        (row.get("diagnostic", "").strip().upper() or "UNKNOWN") for row in rows
    )

    return {
        "dataset_dir": str(dataset_dir.resolve()),
        "metadata_file": str(metadata_path.resolve()),
        "images_dir": str(images_dir.resolve()),
        "metadata_rows": len(rows),
        "metadata_unique_img_ids": len(metadata_img_id_set),
        "png_files_found": len(image_files),
        "duplicate_img_ids_count": len(duplicate_img_ids),
        "missing_images_from_metadata_count": len(missing_images),
        "orphan_images_not_in_metadata_count": len(orphan_images),
        "diagnostic_class_counts": dict(sorted(diagnostic_counts.items())),
        "sample_duplicate_img_ids": duplicate_img_ids[:10],
        "sample_missing_images": missing_images[:10],
        "sample_orphan_images": orphan_images[:10],
    }


def _log_report(report: dict[str, Any], logger: logging.Logger) -> None:
    """Log a concise integrity report."""
    logger.info("PAD-UFES-20 validation report")
    logger.info("metadata_rows=%s", report["metadata_rows"])
    logger.info("metadata_unique_img_ids=%s", report["metadata_unique_img_ids"])
    logger.info("png_files_found=%s", report["png_files_found"])
    logger.info("duplicate_img_ids_count=%s", report["duplicate_img_ids_count"])
    logger.info(
        "missing_images_from_metadata_count=%s",
        report["missing_images_from_metadata_count"],
    )
    logger.info(
        "orphan_images_not_in_metadata_count=%s",
        report["orphan_images_not_in_metadata_count"],
    )
    logger.info("diagnostic_class_counts=%s", report["diagnostic_class_counts"])


def main() -> int:
    """Run PAD-UFES-20 validation and return process exit code."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )
    logger = logging.getLogger("validate_pad_ufes20")

    dataset_dir = args.dataset_dir
    metadata_path = dataset_dir / args.metadata_file
    images_dir = dataset_dir / args.images_subdir

    if not dataset_dir.exists():
        logger.error("Dataset directory does not exist: %s", dataset_dir)
        return 1
    if not metadata_path.exists():
        logger.error("Metadata CSV not found: %s", metadata_path)
        return 1
    if not images_dir.exists():
        logger.error("Images directory not found: %s", images_dir)
        return 1

    report = _build_report(dataset_dir=dataset_dir, metadata_path=metadata_path, images_dir=images_dir)
    _log_report(report, logger)

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        logger.info("Saved report JSON to: %s", args.output_json)

    has_integrity_error = (
        report["missing_images_from_metadata_count"] > 0
        or report["orphan_images_not_in_metadata_count"] > 0
    )
    if has_integrity_error:
        logger.error("Validation failed due to metadata/image mismatch.")
        return 2

    logger.info("Validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
