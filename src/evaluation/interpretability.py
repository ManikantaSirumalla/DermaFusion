"""Interpretability helpers (GradCAM and attention-rollout placeholders)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

__all__ = [
    "attention_rollout",
    "generate_gradcam",
    "generate_interpretation_report",
    "visualize_gradcam",
]


def generate_gradcam(
    model: torch.nn.Module,
    image: torch.Tensor,
    target_class: int,
    layer: torch.nn.Module | None = None,
) -> np.ndarray:
    """Generate a simple GradCAM-like heatmap."""
    _ = layer
    model.eval()
    image = image.unsqueeze(0).clone().detach().requires_grad_(True)
    logits = model({"image": image}) if isinstance(image, torch.Tensor) else model(image)
    score = logits[0, target_class]
    score.backward()
    grad = image.grad.detach().abs().mean(dim=1).squeeze(0)
    heatmap = grad.cpu().numpy()
    heatmap = heatmap / (heatmap.max() + 1e-12)
    return heatmap


def attention_rollout(model: torch.nn.Module, image: torch.Tensor) -> np.ndarray:
    """Fallback attention rollout map."""
    _ = model
    arr = image.detach().cpu().numpy()
    if arr.ndim == 3:
        arr = np.mean(arr, axis=0)
    arr = arr - arr.min()
    arr = arr / (arr.max() + 1e-12)
    return arr


def visualize_gradcam(image: np.ndarray, heatmap: np.ndarray, save_path: str | Path) -> None:
    """Overlay GradCAM heatmap and save image."""
    plt.figure(figsize=(4, 4))
    plt.imshow(image)
    plt.imshow(heatmap, cmap="jet", alpha=0.35)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def generate_interpretation_report(
    model: torch.nn.Module,
    dataset,
    save_dir: str | Path,
    n_samples: int = 5,
) -> None:
    """Generate GradCAM overlays for sample predictions."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for idx in range(min(n_samples, len(dataset))):
        sample = dataset[idx]
        image = sample["image"]
        label = int(sample["label"])
        heatmap = generate_gradcam(model, image, target_class=label)
        img_np = image.detach().cpu().permute(1, 2, 0).numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() + 1e-12)
        visualize_gradcam(img_np, heatmap, save_dir / f"sample_{idx}.png")
