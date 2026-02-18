"""Gradio demo for skin cancer inference."""

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import Any

import gradio as gr
import numpy as np
from PIL import Image
import torch
from omegaconf import OmegaConf

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.transforms import get_val_transforms
from src.evaluation.interpretability import attention_rollout, generate_gradcam
from src.models.model_factory import build_model
from src.utils.config import compose_config
from src.utils.reproducibility import get_device

CLASS_NAMES = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
SEX_TO_IDX = {"male": 0, "female": 1, "unknown": 2}
LOCALIZATION_VALUES = [
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
LOC_TO_IDX = {name: idx for idx, name in enumerate(LOCALIZATION_VALUES)}


def _cfg_value(cfg: Any, key_paths: list[str], default: Any) -> Any:
    for path in key_paths:
        value = OmegaConf.select(cfg, path, default=None)
        if value is not None:
            return value
    return default


def _resolve_checkpoint_path() -> Path | None:
    """Find a checkpoint path from env var or common output locations."""
    env_value = os.environ.get("DERMAFUSION_CKPT", "").strip()
    if env_value:
        env_path = Path(env_value).expanduser()
        if env_path.exists():
            return env_path

    candidates = [
        PROJECT_ROOT / "outputs/checkpoints/best.ckpt",
        PROJECT_ROOT / "outputs/checkpoints/best.pt",
        PROJECT_ROOT / "outputs/checkpoints/best.pth",
    ]
    for path in candidates:
        if path.exists():
            return path

    checkpoint_dir = PROJECT_ROOT / "outputs/checkpoints"
    if checkpoint_dir.exists():
        for pattern in ("*.ckpt", "*.pt", "*.pth"):
            matches = sorted(checkpoint_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def _load_model() -> tuple[torch.nn.Module, Any, torch.device, str]:
    cfg = compose_config(config_dir=PROJECT_ROOT / "configs")
    device = get_device()
    model = build_model(cfg).to(device)
    checkpoint_note = "No trained checkpoint found; using randomly initialized weights."
    ckpt_path = _resolve_checkpoint_path()
    if ckpt_path is not None:
        payload = torch.load(ckpt_path, map_location=device)
        state = payload.get("model_state", payload)
        model.load_state_dict(state, strict=False)
        checkpoint_note = f"Loaded checkpoint: {ckpt_path}"
    model.eval()
    return model, cfg, device, checkpoint_note


MODEL, CFG, DEVICE, CHECKPOINT_NOTE = _load_model()
IMAGE_SIZE = int(_cfg_value(CFG, ["data.data.image_size", "data.image_size"], 380))
USE_METADATA = bool(_cfg_value(CFG, ["model.model.use_metadata", "model.use_metadata"], False))
VAL_TRANSFORM = get_val_transforms(image_size=IMAGE_SIZE)


def _encode_metadata(age: float, sex: str, localization: str) -> torch.Tensor:
    age_norm = float(max(0.0, min(100.0, age)) / 100.0)
    sex_idx = float(SEX_TO_IDX.get(sex, 2))
    loc_idx = float(LOC_TO_IDX.get(localization, LOC_TO_IDX["unknown"]))
    return torch.tensor([age_norm, sex_idx, loc_idx, 0.0, 0.0, 0.0], dtype=torch.float32)


def _heatmap_to_overlay(image_np: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heat = np.clip(heatmap, 0.0, 1.0)
    if heat.shape != image_np.shape[:2]:
        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
            (image_np.shape[1], image_np.shape[0])
        )
        heat = np.asarray(heat_img).astype(np.float32) / 255.0
    overlay = image_np.astype(np.float32) / 255.0
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.5 * heat, 0.0, 1.0)
    return (overlay * 255.0).astype(np.uint8)


def predict(image: Image.Image, age: float, sex: str, localization: str):
    """Run inference for demo UI."""
    image_np = np.asarray(image.convert("RGB"))
    transformed = VAL_TRANSFORM(image=image_np)["image"].unsqueeze(0).to(DEVICE)

    batch = {"image": transformed}
    if USE_METADATA:
        batch["metadata"] = _encode_metadata(age, sex, localization).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = MODEL(batch)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    prob_dict = {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}

    target_class = int(np.argmax(probs))
    if USE_METADATA:
        heatmap = attention_rollout(MODEL, transformed.squeeze(0).cpu())
    else:
        # Temporarily enable gradients for GradCAM generation on model device.
        MODEL.zero_grad(set_to_none=True)
        with torch.enable_grad():
            heatmap = generate_gradcam(MODEL, transformed.squeeze(0), target_class=target_class)
    overlay = _heatmap_to_overlay(image_np, heatmap)

    confidence_note = (
        "High confidence" if float(np.max(probs)) > 0.75 else "Moderate/low confidence; review clinically."
    )
    return (
        prob_dict,
        overlay,
        (
            f"{confidence_note}\n"
            f"{CHECKPOINT_NOTE}\n\n"
            "⚠️ Research-use only. Not a standalone diagnostic tool."
        ),
    )


demo = gr.Interface(
    fn=predict,
    inputs=[
        gr.Image(type="pil", label="Dermatoscopic Image"),
        gr.Slider(0, 100, label="Patient Age", value=50),
        gr.Dropdown(["male", "female", "unknown"], label="Sex", value="unknown"),
        gr.Dropdown(LOCALIZATION_VALUES, label="Lesion Location", value="unknown"),
    ],
    outputs=[
        gr.Label(num_top_classes=7, label="Diagnosis Probabilities"),
        gr.Image(label="GradCAM / Attention Overlay"),
        gr.Textbox(label="Confidence and Disclaimer"),
    ],
    title="Multi-Modal Skin Cancer Classifier",
    description="Upload a dermatoscopic image and optionally provide metadata.",
    article="⚠️ DISCLAIMER: Research-only tool. Not for standalone clinical diagnosis.",
)


if __name__ == "__main__":
    demo.launch()
