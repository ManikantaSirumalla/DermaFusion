"""Quick parity check between bundle runtime and checkpoint runtime."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.inference_runtime import load_inference_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare bundle vs checkpoint inference probabilities.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "preprocessed" / "images" / "val",
        help="Directory containing test images.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Bundle path (optional; auto-discovery checks V2/demo and legacy locations).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best.ckpt",
        help="Checkpoint path.",
    )
    parser.add_argument("--samples", type=int, default=32, help="Number of random images.")
    return parser.parse_args()


def _collect_images(image_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        paths.extend(sorted(image_dir.glob(ext)))
    return paths


def main() -> None:
    args = parse_args()
    images = _collect_images(args.image_dir)
    if not images:
        raise FileNotFoundError(f"No images found in {args.image_dir}")

    bundle_rt = load_inference_runtime(
        project_root=PROJECT_ROOT,
        bundle_path=args.bundle,
        checkpoint_path=args.checkpoint,
    )
    ckpt_rt = load_inference_runtime(
        project_root=PROJECT_ROOT,
        bundle_path=Path("__missing_bundle__.pt"),
        checkpoint_path=args.checkpoint,
    )

    # Parity check only makes sense when class count/order matches.
    if bundle_rt.class_names != ckpt_rt.class_names:
        print("Class mappings differ; parity check skipped.")
        print("bundle classes:", bundle_rt.class_names)
        print("ckpt classes:", ckpt_rt.class_names)
        return

    sample_paths = random.sample(images, k=min(args.samples, len(images)))
    max_abs_list: list[float] = []
    l1_list: list[float] = []
    for path in sample_paths:
        image = Image.open(path).convert("RGB")
        p_bundle = bundle_rt.predict_proba(bundle_rt.preprocess_image(image))
        p_ckpt = ckpt_rt.predict_proba(ckpt_rt.preprocess_image(image))
        delta = np.abs(p_bundle - p_ckpt)
        max_abs_list.append(float(delta.max()))
        l1_list.append(float(delta.mean()))

    print(f"Checked {len(sample_paths)} images")
    print(f"Mean |Δ|: {np.mean(l1_list):.6f}")
    print(f"Max  |Δ|: {np.max(max_abs_list):.6f}")


if __name__ == "__main__":
    main()
