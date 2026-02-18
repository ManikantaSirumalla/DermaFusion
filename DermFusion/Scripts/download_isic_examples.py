"""Download curated ISIC images for offline educational bundles."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from urllib.request import urlopen


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Download curated ISIC example images.")
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to curated JSON manifest produced by curate_isic_education.py",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("DermFusion/Resources/Education"),
        help="Base output directory for downloaded images.",
    )
    parser.add_argument("--max-images", type=int, default=8, help="Maximum images to download.")
    return parser.parse_args()


def select_license_allowed(records: list[dict], max_images: int) -> list[dict]:
    """Filters records to preferred educational licenses."""
    allowed = {"CC-0", "CC-BY", "CC-BY-NC"}
    filtered = []
    for record in records:
        license_name = str(record.get("license", "")).upper()
        if any(item in license_name for item in allowed):
            filtered.append(record)
        if len(filtered) >= max_images:
            break
    return filtered


def main() -> None:
    """Downloads curated ISIC image URLs into local education folders."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    payload = json.loads(args.manifest.read_text())
    diagnosis = payload.get("diagnosis", "unknown")
    images = select_license_allowed(payload.get("images", []), max_images=args.max_images)

    target_dir = args.output_dir / diagnosis
    target_dir.mkdir(parents=True, exist_ok=True)

    for index, image in enumerate(images, start=1):
        image_url = image.get("imageUrl") or image.get("url")
        if not image_url:
            continue
        destination = target_dir / f"{diagnosis}_{index:02d}.jpg"
        with urlopen(image_url) as response:  # noqa: S310 (trusted public source)
            destination.write_bytes(response.read())
        logging.info("Downloaded %s", destination)

    logging.info("Finished downloading %d images for diagnosis=%s", len(images), diagnosis)


if __name__ == "__main__":
    main()
