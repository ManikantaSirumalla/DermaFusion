"""Export inference bundle from trained checkpoint.

This creates a Colab-compatible `dermafusion_bundle.pt` artifact consumed by
CLI/Gradio runtime loaders.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.models.model_factory import build_model
from src.utils.config import compose_config
from omegaconf import OmegaConf

DEFAULT_CLASS_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export deployment bundle from best checkpoint.")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs/checkpoints/best.ckpt",
        help="Checkpoint path (.ckpt/.pt/.pth).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs/export/dermafusion_bundle.pt",
        help="Output bundle path.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs",
        help="Hydra config directory.",
    )
    parser.add_argument(
        "--mel-threshold",
        type=float,
        default=0.30,
        help="Melanoma probability threshold for deployment.",
    )
    parser.add_argument(
        "--require-selection-metric",
        type=str,
        default="selection_score",
        help=(
            "Checkpoint metric key required for top-model validation "
            "(empty string disables the requirement)."
        ),
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=380,
        help="Inference image size used by runtime preprocessing.",
    )
    parser.add_argument(
        "--class-order",
        type=str,
        default=",".join(DEFAULT_CLASS_ORDER),
        help="Comma-separated class order matching model logits.",
    )
    return parser.parse_args()


def _extract_state_dict(payload: object) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model_state"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise ValueError("Unable to extract state_dict from checkpoint payload.")


def main() -> None:
    args = parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    cfg = compose_config(config_dir=args.config_dir)
    model = build_model(cfg)
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    checkpoint_metrics = payload.get("metrics", {}) if isinstance(payload, dict) else {}
    require_metric = args.require_selection_metric.strip().lower()
    if require_metric and isinstance(checkpoint_metrics, dict) and require_metric not in checkpoint_metrics:
        raise KeyError(
            f"Missing required checkpoint metric '{require_metric}'. "
            "Train using selection_metric=val_score so this key is stored."
        )
    if isinstance(checkpoint_metrics, dict):
        checkpoint_epoch = payload.get("epoch", None)
    else:
        checkpoint_epoch = None

    class_names = [name.strip() for name in args.class_order.split(",") if name.strip()]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    num_classes = len(class_names)
    backbone = str(
        OmegaConf.select(cfg, "model.model.backbone", default=None)
        or OmegaConf.select(cfg, "model.backbone", default="efficientnet_b4")
    )
    use_metadata_cfg = OmegaConf.select(cfg, "model.model.use_metadata", default=None)
    if use_metadata_cfg is None:
        use_metadata_cfg = OmegaConf.select(cfg, "model.use_metadata", default=False)
    use_metadata = bool(use_metadata_cfg)

    bundle = {
        "model_loader": "dermafusion",
        "model_name": backbone,
        "num_classes": int(num_classes),
        "class_to_idx": class_to_idx,
        "img_size": int(args.image_size),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "mel_threshold": float(args.mel_threshold),
        "use_metadata": use_metadata,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_metrics": checkpoint_metrics,
        "checkpoint_epoch": checkpoint_epoch,
        "config": None,
        "state_dict": model.state_dict(),
    }

    # Store plain dict config (OmegaConf objects are not always portable).
    try:
        bundle["config"] = OmegaConf.to_container(cfg, resolve=True)
    except Exception:
        bundle["config"] = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, args.output)
    print(f"Saved bundle: {args.output}")


if __name__ == "__main__":
    main()
