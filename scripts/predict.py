"""CLI inference for single-image prediction."""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.inference_runtime import load_inference_runtime
from src.utils.logging import setup_logger

LOC_VALUES = [
    "back",
    "lower extremity",
    "trunk",
    "upper extremity",
    "abdomen",
    "face",
    "chest",
    "foot",
    "neck",
    "scalp",
    "hand",
    "ear",
    "genital",
    "unknown",
]


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Single-image skin lesion prediction.")
    parser.add_argument("--image", type=Path, required=True, help="Path to dermoscopy image.")
    parser.add_argument("--age", type=float, default=50.0, help="Patient age.")
    parser.add_argument(
        "--sex",
        type=str,
        default="unknown",
        choices=["male", "female", "unknown"],
        help="Patient sex category.",
    )
    parser.add_argument(
        "--localization",
        type=str,
        default="unknown",
        choices=LOC_VALUES,
        help="Lesion localization category.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/checkpoints/best.ckpt",
        help="Checkpoint path.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Bundle path (optional; auto-discovery checks V2/demo and legacy locations).",
    )
    return parser.parse_args()


def main() -> None:
    """Run one-shot inference and print probabilities."""
    args = parse_args()
    logger = setup_logger("predict")
    if not args.image.exists():
        raise FileNotFoundError(f"Image path not found: {args.image}")

    runtime = load_inference_runtime(
        project_root=PROJECT_ROOT,
        bundle_path=args.bundle,
        checkpoint_path=args.checkpoint,
    )
    logger.info(runtime.checkpoint_note)

    image = Image.open(args.image).convert("RGB")
    image_batch = runtime.preprocess_image(image)
    metadata_batch = None
    if runtime.use_metadata:
        metadata_batch = runtime.encode_metadata(
            age=args.age,
            sex=args.sex,
            localization=args.localization,
        ).unsqueeze(0)

    probs = runtime.predict_proba(image_batch=image_batch, metadata_batch=metadata_batch)
    prob_dict = runtime.probabilities_dict(probs)
    ranked = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    mel_prob, high_risk = runtime.melanoma_decision(probs)

    logger.info("Prediction ranking:")
    for name, prob in ranked:
        logger.info("  %s: %.4f", name, prob)
    logger.info(
        "MEL probability: %.4f (threshold=%.2f) => %s",
        mel_prob,
        runtime.mel_threshold,
        "HIGH RISK" if high_risk else "lower risk",
    )


if __name__ == "__main__":
    main()
