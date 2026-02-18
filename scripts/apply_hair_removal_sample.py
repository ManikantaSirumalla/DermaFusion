"""Apply hair removal (DullRazor) to the sample comparison image and save for viewing."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import dull_razor, shades_of_gray

OUT_DIR = PROJECT_ROOT / "outputs" / "sample_comparison"
IMAGE_SIZE = 380


def main() -> None:
    raw_path = PROJECT_ROOT / "data" / "raw" / "images" / "train" / "ISIC_0024306.jpg"
    if not raw_path.exists():
        print("Sample image not found. Ensure data/raw/images/train/ISIC_0024306.jpg exists.")
        sys.exit(1)

    with Image.open(raw_path) as img:
        rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.uint8)

    # Pipeline: color constancy -> hair removal -> resize
    arr = shades_of_gray(arr)
    arr = dull_razor(arr)

    h, w = arr.shape[:2]
    if h != IMAGE_SIZE or w != IMAGE_SIZE:
        resized = Image.fromarray(arr).resize(
            (IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS
        )
        arr = np.asarray(resized, dtype=np.uint8)

    out_path = OUT_DIR / "preprocessed_hair_removed_ISIC_0024306.jpg"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(out_path, "JPEG", quality=95)
    print(f"Saved: {out_path}")
    print("Compare: raw_ISIC_0024306.jpg | preprocessed_ISIC_0024306.jpg | preprocessed_hair_removed_ISIC_0024306.jpg")


if __name__ == "__main__":
    main()
