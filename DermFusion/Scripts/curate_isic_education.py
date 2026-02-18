"""Curate educational examples from ISIC at build time.

This script is for development-time curation only. It should never be used from
runtime app code.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any
from urllib.parse import urlencode
from urllib.request import urlopen

ISIC_API_BASE = "https://api.isic-archive.com/api/v1/images"


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Curate ISIC educational metadata.")
    parser.add_argument("--diagnosis", required=True, help="Diagnosis filter (example: melanoma).")
    parser.add_argument("--limit", type=int, default=25, help="Maximum records to request.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Scripts/output/isic_curated.json"),
        help="Output JSON path for curated metadata.",
    )
    return parser.parse_args()


def fetch_isic_images(diagnosis: str, limit: int) -> list[dict[str, Any]]:
    """Fetches image metadata from the ISIC API."""
    query = urlencode({"diagnosis": diagnosis, "limit": limit})
    url = f"{ISIC_API_BASE}?{query}"
    with urlopen(url) as response:  # noqa: S310 (static trusted endpoint)
        payload = json.loads(response.read().decode("utf-8"))
    if isinstance(payload, list):
        return payload
    return payload.get("results", [])


def main() -> None:
    """Curates diagnosis-filtered image metadata for offline educational bundling."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    logging.info("Fetching ISIC metadata for diagnosis='%s' (limit=%d).", args.diagnosis, args.limit)
    images = fetch_isic_images(diagnosis=args.diagnosis, limit=args.limit)

    curated = {
        "diagnosis": args.diagnosis,
        "count": len(images),
        "images": images,
    }
    args.output.write_text(json.dumps(curated, indent=2))
    logging.info("Saved curated metadata to %s", args.output)


if __name__ == "__main__":
    main()
