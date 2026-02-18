"""CLI inference for single-image prediction."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_val_transforms
from src.models.model_factory import build_model
from src.utils.config import compose_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import get_device

CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
SEX_TO_IDX = {"male": 0, "female": 1, "unknown": 2}
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
LOC_TO_IDX = {name: idx for idx, name in enumerate(LOC_VALUES)}


def _cfg_value(cfg: Any, key_paths: list[str], default: Any) -> Any:
    for path in key_paths:
        value = OmegaConf.select(cfg, path, default=None)
        if value is not None:
            return value
    return default


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
    return parser.parse_args()


def _encode_metadata(age: float, sex: str, localization: str) -> torch.Tensor:
    age_norm = float(max(0.0, min(100.0, age)) / 100.0)
    return torch.tensor(
        [
            age_norm,
            float(SEX_TO_IDX.get(sex, 2)),
            float(LOC_TO_IDX.get(localization, LOC_TO_IDX["unknown"])),
            0.0,
            0.0,
            0.0,
        ],
        dtype=torch.float32,
    )


def main() -> None:
    """Run one-shot inference and print probabilities."""
    args = parse_args()
    logger = setup_logger("predict")
    if not args.image.exists():
        raise FileNotFoundError(f"Image path not found: {args.image}")

    cfg = compose_config(config_dir=PROJECT_ROOT / "configs")
    device = get_device()
    model = build_model(cfg).to(device)
    if args.checkpoint.exists():
        payload = torch.load(args.checkpoint, map_location=device)
        state = payload.get("model_state", payload)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", args.checkpoint)
    else:
        logger.warning("Checkpoint not found at %s; using current model weights.", args.checkpoint)

    image_size = int(_cfg_value(cfg, ["data.data.image_size", "data.image_size"], 380))
    use_metadata = bool(_cfg_value(cfg, ["model.model.use_metadata", "model.use_metadata"], False))
    transform = get_val_transforms(image_size=image_size)

    image_np = np.asarray(Image.open(args.image).convert("RGB"))
    image_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
    batch: dict[str, torch.Tensor] = {"image": image_tensor}
    if use_metadata:
        batch["metadata"] = _encode_metadata(args.age, args.sex, args.localization).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    ranked = sorted(zip(CLASS_NAMES, probs.tolist()), key=lambda x: x[1], reverse=True)
    logger.info("Prediction ranking:")
    for name, prob in ranked:
        logger.info("  %s: %.4f", name, prob)


if __name__ == "__main__":
    main()
