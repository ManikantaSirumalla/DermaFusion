"""Shared inference runtime for CLI, Gradio, and deployment bundles.

Bundle-first loading keeps local code aligned with Colab-exported artifacts.
Fallback mode still supports legacy checkpoint+Hydra config workflows.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import timm
import torch
from omegaconf import OmegaConf

from src.data.transforms import get_val_transforms
from src.models.model_factory import build_model
from src.utils.config import compose_config
from src.utils.reproducibility import get_device

DEFAULT_CLASS_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
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


def _resolve_checkpoint_path(project_root: Path, explicit_path: Path | None = None) -> Path | None:
    if explicit_path is not None and explicit_path.exists():
        return explicit_path

    candidates = [
        project_root / "outputs/checkpoints/best.ckpt",
        project_root / "outputs/checkpoints/best.pt",
        project_root / "outputs/checkpoints/best.pth",
    ]
    for path in candidates:
        if path.exists():
            return path

    ckpt_dir = project_root / "outputs/checkpoints"
    if ckpt_dir.exists():
        for pattern in ("*.ckpt", "*.pt", "*.pth"):
            matches = sorted(ckpt_dir.glob(pattern))
            if matches:
                return matches[0]
    return None


def _iter_v2_bundle_candidates(project_root: Path) -> list[Path]:
    """Return ordered V2 artifact candidates under demo/dermafusion_new_V2."""
    v2_root = project_root / "demo" / "dermafusion_new_V2"
    if not v2_root.exists():
        return []

    candidates: list[Path] = [
        # Prioritize final curated and threshold-tuned exports first.
        v2_root / "final_t020" / "dermafusion_bundle.pt",
        v2_root / "export_v2" / "dermafusion_bundle_t020.pt",
        v2_root / "export_v2" / "dermafusion_bundle.pt",
    ]
    run_candidates = sorted(
        [path for path in v2_root.glob("run_*/dermafusion_bundle.pt") if path.is_file()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    candidates.extend(run_candidates)
    return candidates


def resolve_bundle_path(project_root: Path, explicit_path: Path | None = None) -> Path | None:
    """Resolve deployment bundle path from explicit/env/V2/default candidate locations."""
    candidates: list[Path] = []
    if explicit_path is not None:
        candidates.append(explicit_path)

    bundle_env = os.environ.get("DERMAFUSION_BUNDLE", "").strip()
    if bundle_env:
        candidates.append(Path(bundle_env).expanduser())

    candidates.extend(
        [
            project_root / "app" / "deployment_Bundle" / "dermafusion_ensemble.pt",
            *_iter_v2_bundle_candidates(project_root),
            project_root / "outputs/export/dermafusion_bundle.pt",
            project_root / "dermafusion_bundle.pt",
            # Backward-compatible typo fallback seen in manual exports.
            project_root / "dermaduaion_bundle.pt",
            project_root / "outputs/export/dermaduaion_bundle.pt",
        ]
    )

    for path in candidates:
        if path.exists():
            try:
                return path.resolve()
            except Exception:
                return path
    return None


def _resolve_bundle_path(project_root: Path, explicit_path: Path | None = None) -> Path | None:
    """Backwards-compatible private alias."""
    return resolve_bundle_path(project_root=project_root, explicit_path=explicit_path)


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, dict):
        for key in ("state_dict", "model_state_dict", "model_state"):
            value = payload.get(key)
            if isinstance(value, dict):
                return value
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload  # pure state dict payload
    raise ValueError("Unable to extract model state_dict from payload.")


def _env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


def _normalize_class_to_idx(mapping: dict[str, Any] | None, num_classes: int) -> dict[str, int]:
    if mapping:
        out: dict[str, int] = {}
        for key, value in mapping.items():
            out[str(key)] = int(value)
        return out
    return {name: idx for idx, name in enumerate(DEFAULT_CLASS_ORDER[:num_classes])}


@dataclass
class InferenceRuntime:
    """Loaded inference runtime that exposes shared prediction helpers."""

    model: torch.nn.Module
    device: torch.device
    class_to_idx: dict[str, int]
    image_size: int
    mean: list[float]
    std: list[float]
    mel_threshold: float
    use_metadata: bool
    transform: Any | None = None
    checkpoint_note: str = ""
    source: str = "checkpoint"

    @property
    def class_names(self) -> list[str]:
        return [name for name, _ in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])]

    def encode_metadata(self, age: float, sex: str, localization: str) -> torch.Tensor:
        age_norm = float(max(0.0, min(100.0, age)) / 100.0)
        sex_idx = float(SEX_TO_IDX.get(str(sex).lower(), 2))
        loc_idx = float(LOC_TO_IDX.get(str(localization).lower(), LOC_TO_IDX["unknown"]))
        return torch.tensor([age_norm, sex_idx, loc_idx, 0.0, 0.0, 0.0], dtype=torch.float32)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        image_rgb = image.convert("RGB")
        if self.transform is not None:
            transformed = self.transform(image=np.asarray(image_rgb))["image"]
            return transformed.unsqueeze(0)

        resized = image_rgb.resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        arr = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        mean = torch.tensor(self.mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(self.std, dtype=torch.float32).view(1, 3, 1, 1)
        return (tensor - mean) / std

    def forward_logits(
        self,
        image_batch: torch.Tensor,
        metadata_batch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        image_batch = image_batch.to(self.device, non_blocking=True)
        if metadata_batch is not None:
            metadata_batch = metadata_batch.to(self.device, non_blocking=True)

        # Try project model signature first (dict input), then pure tensor (timm).
        with torch.no_grad():
            if self.use_metadata and metadata_batch is not None:
                batch = {"image": image_batch, "metadata": metadata_batch}
            else:
                batch = {"image": image_batch}
            try:
                return self.model(batch)  # type: ignore[arg-type]
            except Exception:
                return self.model(image_batch)  # type: ignore[arg-type]

    def predict_proba(
        self,
        image_batch: torch.Tensor,
        metadata_batch: torch.Tensor | None = None,
    ) -> np.ndarray:
        self.model.eval()
        logits = self.forward_logits(image_batch=image_batch, metadata_batch=metadata_batch)
        probs = torch.softmax(logits, dim=1).squeeze(0).detach().cpu().numpy()
        return probs.astype(np.float64)

    def probabilities_dict(self, probs: np.ndarray) -> dict[str, float]:
        return {
            name: float(probs[idx])
            for name, idx in sorted(self.class_to_idx.items(), key=lambda kv: kv[1])
            if idx < len(probs)
        }

    def melanoma_decision(self, probs: np.ndarray) -> tuple[float, bool]:
        mel_idx = self.class_to_idx.get("MEL")
        if mel_idx is None:
            mel_idx = self.class_to_idx.get("mel")
        if mel_idx is None or mel_idx >= len(probs):
            return 0.0, False
        mel_prob = float(probs[mel_idx])
        return mel_prob, mel_prob >= float(self.mel_threshold)


def _load_from_bundle(bundle_path: Path, device: torch.device) -> InferenceRuntime:
    # The deployment_Bundle pickle was saved from a Colab __main__ kernel; make
    # the class importable there before unpickling so torch.load can find it.
    from src.deployment.ensemble import register_for_unpickling

    register_for_unpickling()

    payload = torch.load(bundle_path, map_location=device, weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"Bundle must be dict-like, got: {type(payload)}")

    # Pickled-module bundle (deployment_Bundle/dermafusion_ensemble.pt format):
    # the whole nn.Module is stored under "model" and class metadata uses
    # `class_names` / `class_thresholds` instead of `class_to_idx`.
    preloaded = payload.get("model")
    if isinstance(preloaded, torch.nn.Module):
        names = list(payload.get("class_names") or [])
        class_to_idx = {str(n).upper(): i for i, n in enumerate(names)} or _normalize_class_to_idx(
            payload.get("class_to_idx"), int(payload.get("num_classes", len(names) or 8))
        )
        thr_dict = payload.get("class_thresholds")
        mel_threshold = (
            float(thr_dict.get("MEL", 0.30))
            if isinstance(thr_dict, dict)
            else float(payload.get("mel_threshold", 0.30))
        )
        model = preloaded.to(device).eval()
        image_size = int(payload.get("img_size", getattr(preloaded, "img_size", 380)))
        mean = [float(x) for x in payload.get("mean", getattr(preloaded, "mean", [0.485, 0.456, 0.406]))]
        std = [float(x) for x in payload.get("std", getattr(preloaded, "std", [0.229, 0.224, 0.225]))]
        use_metadata = False

        apply_color_constancy = _env_flag("DERMAFUSION_COLOR_CONSTANCY", True)
        transform = get_val_transforms(
            image_size=image_size,
            apply_color_constancy=apply_color_constancy,
        )
        return InferenceRuntime(
            model=model,
            device=device,
            class_to_idx=class_to_idx,
            image_size=image_size,
            mean=mean,
            std=std,
            mel_threshold=mel_threshold,
            use_metadata=use_metadata,
            transform=transform,
            checkpoint_note=f"Loaded bundle: {bundle_path}",
            source="bundle",
        )

    model_name = str(payload.get("model_name", "efficientnet_b4"))
    class_to_idx = _normalize_class_to_idx(payload.get("class_to_idx"), int(payload.get("num_classes", 7)))
    num_classes = int(payload.get("num_classes", max(class_to_idx.values()) + 1))
    image_size = int(payload.get("img_size", 380))
    mean = [float(x) for x in payload.get("mean", [0.485, 0.456, 0.406])]
    std = [float(x) for x in payload.get("std", [0.229, 0.224, 0.225])]
    mel_threshold = float(payload.get("mel_threshold", 0.30))
    use_metadata = bool(payload.get("use_metadata", False))

    model_loader = str(payload.get("model_loader", "timm")).lower()
    if model_loader == "dermafusion":
        cfg_dict = payload.get("config")
        if cfg_dict is None:
            raise ValueError("dermafusion bundle requires embedded `config`.")
        cfg = OmegaConf.create(cfg_dict)
        # Bundle already includes trained weights; disable backbone pretrain download at load time.
        if OmegaConf.select(cfg, "model.model.pretrained", default=None) is not None:
            OmegaConf.update(cfg, "model.model.pretrained", False, force_add=False)
        if OmegaConf.select(cfg, "model.pretrained", default=None) is not None:
            OmegaConf.update(cfg, "model.pretrained", False, force_add=False)
        model = build_model(cfg)
    elif model_loader == "ensemble":
        # Lightweight state-dict-only ensemble bundle (alternative format).
        from src.deployment.ensemble import DermaFusionEnsemble

        member_dicts = payload.get("members") or []
        if not member_dicts:
            raise ValueError("ensemble bundle requires non-empty `members` list.")
        backbones = [str(m["backbone"]) for m in member_dicts]
        weights = [float(m.get("weight", 1.0)) for m in member_dicts]
        names = list(class_to_idx.keys())
        model = DermaFusionEnsemble(
            backbones=backbones,
            weights=weights,
            class_names=names,
            mean=mean,
            std=std,
            img_size=image_size,
        )
    else:
        model = timm.create_model(model_name, pretrained=False, num_classes=num_classes)

    state_dict = _extract_state_dict(payload)
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Bundle exports typically omit preprocessing config.
    # Default to color constancy to match dermoscopy validation preprocessing.
    apply_color_constancy = _env_flag("DERMAFUSION_COLOR_CONSTANCY", True)
    transform = get_val_transforms(
        image_size=image_size,
        apply_color_constancy=apply_color_constancy,
    )

    return InferenceRuntime(
        model=model,
        device=device,
        class_to_idx=class_to_idx,
        image_size=image_size,
        mean=mean,
        std=std,
        mel_threshold=mel_threshold,
        use_metadata=use_metadata,
        transform=transform,
        checkpoint_note=f"Loaded bundle: {bundle_path}",
        source="bundle",
    )


def _load_from_checkpoint(
    project_root: Path,
    device: torch.device,
    checkpoint_path: Path | None = None,
    config_overrides: list[str] | None = None,
) -> InferenceRuntime:
    cfg = compose_config(config_dir=project_root / "configs", overrides=config_overrides or [])
    model = build_model(cfg).to(device)
    model.eval()

    ckpt_path = _resolve_checkpoint_path(project_root=project_root, explicit_path=checkpoint_path)
    note = "No trained checkpoint found; using randomly initialized weights."
    if ckpt_path is not None and ckpt_path.exists():
        payload = torch.load(ckpt_path, map_location=device, weights_only=False)
        state = _extract_state_dict(payload)
        model.load_state_dict(state, strict=False)
        note = f"Loaded checkpoint: {ckpt_path}"

    image_size = int(_cfg_value(cfg, ["data.data.image_size", "data.image_size"], 380))
    use_preprocessed = bool(_cfg_value(cfg, ["data.data.use_preprocessed", "data.use_preprocessed"], False))
    use_metadata = bool(_cfg_value(cfg, ["model.model.use_metadata", "model.use_metadata"], False))
    transform = get_val_transforms(image_size=image_size, apply_color_constancy=not use_preprocessed)
    class_to_idx = {name: idx for idx, name in enumerate(DEFAULT_CLASS_ORDER)}

    return InferenceRuntime(
        model=model,
        device=device,
        class_to_idx=class_to_idx,
        image_size=image_size,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        mel_threshold=0.30,
        use_metadata=use_metadata,
        transform=transform,
        checkpoint_note=note,
        source="checkpoint",
    )


def load_inference_runtime(
    project_root: Path,
    bundle_path: Path | None = None,
    checkpoint_path: Path | None = None,
    device_override: str | None = None,
    config_overrides: list[str] | None = None,
) -> InferenceRuntime:
    """Load bundle-first inference runtime with checkpoint fallback."""
    device = get_device(device_override)

    resolved_bundle = resolve_bundle_path(project_root=project_root, explicit_path=bundle_path)
    if resolved_bundle is not None:
        return _load_from_bundle(bundle_path=resolved_bundle, device=device)

    return _load_from_checkpoint(
        project_root=project_root,
        device=device,
        checkpoint_path=checkpoint_path,
        config_overrides=config_overrides,
    )
