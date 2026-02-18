"""Build a preprocessed image dataset for faster, consistent training.

Reads raw images from data/raw/images/{train,val,test}, applies:
- RGB conversion
- Shades of Gray color constancy (dermoscopy-standard)
- Optional DullRazor-style hair removal (no-op stub unless implemented)
- Resize to a fixed size (default 380x380 for EfficientNet-B4)
- Saves as JPEG to data/preprocessed/images/{train,val,test}

Training can then use data.data.use_preprocessed=true and skip color constancy
in the transform pipeline. Metadata stays in data/raw/metadata (unchanged).
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

from src.data.preprocessing import dull_razor, shades_of_gray

SPLITS = ("train", "val", "test")
JPEG_QUALITY = 95


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess raw dermoscopy images (color constancy + resize)."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "images",
        help="Root containing train/val/test subdirs of raw images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed" / "images",
        help="Output root; train/val/test subdirs will be created.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=380,
        help="Target size (height and width) for resized images.",
    )
    parser.add_argument(
        "--apply-hair-removal",
        action="store_true",
        help="Apply DullRazor-style hair removal (currently no-op unless implemented).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Parallel workers (0 = sequential).",
    )
    return parser.parse_args()


def process_one(
    raw_path: Path,
    out_path: Path,
    image_size: int,
    apply_hair: bool,
) -> bool:
    """Load, preprocess, and save one image. Returns True on success."""
    try:
        with Image.open(raw_path) as img:
            rgb = img.convert("RGB")
        arr = np.asarray(rgb, dtype=np.uint8)
        arr = shades_of_gray(arr)
        if apply_hair:
            arr = dull_razor(arr)
        h, w = arr.shape[:2]
        if h != image_size or w != image_size:
            resized = Image.fromarray(arr).resize(
                (image_size, image_size), Image.Resampling.LANCZOS
            )
            arr = np.asarray(resized, dtype=np.uint8)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(out_path, "JPEG", quality=JPEG_QUALITY)
        return True
    except Exception:
        return False


def run_sequential(
    raw_root: Path,
    output_root: Path,
    image_size: int,
    apply_hair: bool,
    logger: logging.Logger,
) -> tuple[int, int]:
    total, failed = 0, 0
    for split in SPLITS:
        raw_dir = raw_root / split
        out_dir = output_root / split
        if not raw_dir.exists():
            logger.warning("Skip %s: %s does not exist", split, raw_dir)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for raw_path in sorted(raw_dir.glob("*.jpg")) + sorted(raw_dir.glob("*.jpeg")) + sorted(raw_dir.glob("*.png")):
            total += 1
            out_path = out_dir / f"{raw_path.stem}.jpg"
            if process_one(raw_path, out_path, image_size, apply_hair):
                if total % 1000 == 0:
                    logger.info("Processed %d images (split=%s)...", total, split)
            else:
                failed += 1
                logger.warning("Failed: %s", raw_path)
    return total, failed


def main() -> None:
    args = parse_args()
    log_file = PROJECT_ROOT / "outputs" / "logs" / "preprocess_data.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("preprocess_data")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    logger.addHandler(logging.StreamHandler())
    logger.addHandler(logging.FileHandler(log_file))
    for h in logger.handlers:
        h.setFormatter(fmt)

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw directory not found: {args.raw_dir}")

    logger.info("Raw root: %s", args.raw_dir)
    logger.info("Output root: %s", args.output_dir)
    logger.info("Image size: %d", args.image_size)

    total, failed = run_sequential(
        args.raw_dir,
        args.output_dir,
        args.image_size,
        args.apply_hair_removal,
        logger,
    )
    logger.info("Done. Total=%d, failed=%d", total, failed)
    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
