"""Apply DullRazor hair removal to all preprocessed images for training.

Reads from data/preprocessed/images/{train,val,test}, applies dull_razor only
(images are already color-corrected and resized), saves to
data/preprocessed_hair_removed/images/{train,val,test}.

Then train with:
  data.data.use_preprocessed=true \\
  data.data.preprocessed_image_dir=data/preprocessed_hair_removed/images
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import dull_razor

SPLITS = ("train", "val", "test")
JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply hair removal to preprocessed images (DullRazor)."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed" / "images",
        help="Root containing train/val/test of preprocessed images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed_hair_removed" / "images",
        help="Output root; train/val/test subdirs will be created.",
    )
    return parser.parse_args()


def process_one(in_path: Path, out_path: Path) -> bool:
    """Load image, apply DullRazor, save. Returns True on success."""
    try:
        with Image.open(in_path) as img:
            rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        arr = dull_razor(arr)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out_path, "JPEG", quality=JPEG_QUALITY)
        return True
    except Exception:
        return False


def main() -> None:
    args = parse_args()
    log_file = PROJECT_ROOT / "outputs" / "logs" / "apply_hair_removal_to_preprocessed.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("hair_removal")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_file))
    for h in logger.handlers:
        h.setFormatter(fmt)

    if not args.input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    logger.info("Input:  %s", args.input_dir)
    logger.info("Output: %s", args.output_dir)

    total, failed = 0, 0
    for split in SPLITS:
        in_dir = args.input_dir / split
        out_dir = args.output_dir / split
        if not in_dir.exists():
            logger.warning("Skip %s: %s does not exist", split, in_dir)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for in_path in sorted(in_dir.glob("*.jpg")) + sorted(in_dir.glob("*.jpeg")) + sorted(in_dir.glob("*.png")):
            total += 1
            out_path = out_dir / f"{in_path.stem}.jpg"
            if process_one(in_path, out_path):
                if total % 1000 == 0:
                    logger.info("Processed %d images (split=%s)...", total, split)
            else:
                failed += 1
                logger.warning("Failed: %s", in_path)

    logger.info("Done. Total=%d, failed=%d", total, failed)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
