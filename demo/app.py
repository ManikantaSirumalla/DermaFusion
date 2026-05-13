"""Gradio demo for DermaFusion skin lesion inference."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
_CACHE_ROOT = Path(os.environ.get("DERMAFUSION_CACHE_DIR", "/tmp/dermafusion-cache"))
_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
(_CACHE_ROOT / "matplotlib").mkdir(parents=True, exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "matplotlib"))

import gradio as gr


DEFAULT_IMAGE_SIZE = 380
DEFAULT_AGE = 50
DEFAULT_CLASS_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
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
SEX_VALUES = ["unknown", "female", "male"]
DISCLAIMER = (
    "Research-use only. This demo is not a standalone diagnostic tool and does not "
    "replace clinician review, biopsy, or emergency care for concerning lesions."
)
CLASS_DISPLAY_NAMES = {
    "mel": "Melanoma",
    "nv": "Nevus",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bkl": "Benign keratosis-like lesion",
    "df": "Dermatofibroma",
    "vasc": "Vascular lesion",
}


@dataclass(frozen=True)
class PredictionArtifacts:
    """Display-ready values emitted by the inference flow."""

    probabilities: dict[str, float]
    overlay: np.ndarray
    summary_markdown: str
    model_status_markdown: str


def _env_path(name: str) -> Path | None:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else None


def _env_flag(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _available_port(host: str) -> int:
    import socket

    bind_host = "127.0.0.1" if host in {"0.0.0.0", "::"} else host
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((bind_host, 0))
        return int(sock.getsockname()[1])


def _extend_gradio_start_timeout(timeout_seconds: float = 30.0) -> None:
    import threading
    import time

    from gradio import networking
    from gradio.exceptions import ServerFailedToStartError

    if getattr(networking.Server.run_in_thread, "_dermafusion_patched", False):
        return

    def run_in_thread(self: Any) -> None:
        self.thread = threading.Thread(target=self.run, daemon=True)
        if self.reloader:
            self.watch_thread = threading.Thread(target=self.watch, daemon=True)
            self.watch_thread.start()
        self.thread.start()
        start = time.time()
        while not self.started:
            time.sleep(1e-3)
            if time.time() - start > timeout_seconds:
                raise ServerFailedToStartError(
                    "Server failed to start. Please check that the port is available."
                )

    run_in_thread._dermafusion_patched = True  # type: ignore[attr-defined]
    networking.Server.run_in_thread = run_in_thread


def _relax_gradio_local_url_check() -> None:
    from gradio import networking

    if getattr(networking.url_ok, "_dermafusion_patched", False):
        return

    original_url_ok = networking.url_ok

    def url_ok(url: str) -> bool:
        if url.startswith(("http://127.0.0.1:", "http://localhost:")):
            return True
        return original_url_ok(url)

    url_ok._dermafusion_patched = True  # type: ignore[attr-defined]
    networking.url_ok = url_ok


def _class_label(class_name: str) -> str:
    key = str(class_name).lower()
    readable = CLASS_DISPLAY_NAMES.get(key, str(class_name).upper())
    return f"{readable} ({str(class_name).upper()})"


def _empty_probabilities(class_names: list[str] | None = None) -> dict[str, float]:
    names = class_names or DEFAULT_CLASS_ORDER
    return {_class_label(name): 0.0 for name in names}


def _empty_overlay(size: int = DEFAULT_IMAGE_SIZE) -> np.ndarray:
    return np.zeros((size, size, 3), dtype=np.uint8)


def _normalise_probabilities(runtime: Any, probs: np.ndarray) -> dict[str, float]:
    raw = runtime.probabilities_dict(probs)
    return {_class_label(name): float(value) for name, value in raw.items()}


def _model_source_note(runtime: Any) -> str:
    metadata_note = "enabled" if runtime.use_metadata else "not used by this loaded model"
    return (
        f"Loaded `{runtime.source}` runtime on `{runtime.device}`. "
        f"Image size: {runtime.image_size}px. Metadata: {metadata_note}."
    )


def _quick_bundle_candidate() -> Path | None:
    candidates: list[Path] = []
    env_bundle = _env_path("DERMAFUSION_BUNDLE")
    if env_bundle is not None:
        candidates.append(env_bundle)

    v2_root = PROJECT_ROOT / "demo" / "dermafusion_new_V2"
    candidates.extend(
        [
            v2_root / "final_t020" / "dermafusion_bundle.pt",
            v2_root / "export_v2" / "dermafusion_bundle_t020.pt",
            v2_root / "export_v2" / "dermafusion_bundle.pt",
        ]
    )
    if v2_root.exists():
        candidates.extend(
            sorted(
                [path for path in v2_root.glob("run_*/dermafusion_bundle.pt") if path.is_file()],
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
        )
    candidates.extend(
        [
            PROJECT_ROOT / "outputs/export/dermafusion_bundle.pt",
            PROJECT_ROOT / "dermafusion_bundle.pt",
            PROJECT_ROOT / "dermaduaion_bundle.pt",
            PROJECT_ROOT / "outputs/export/dermaduaion_bundle.pt",
        ]
    )

    for path in candidates:
        if path.exists():
            return path
    return None


def _initial_status_markdown() -> str:
    resolved_bundle = _quick_bundle_candidate()
    checkpoint_path = _env_path("DERMAFUSION_CKPT") or PROJECT_ROOT / "outputs/checkpoints/best.ckpt"
    artifact_note = (
        f"Bundle candidate: `{resolved_bundle}`"
        if resolved_bundle is not None
        else f"No bundle found yet. Checkpoint fallback: `{checkpoint_path}`"
    )
    return (
        "### Model status\n"
        "The model loads once on first analysis or warm-up.\n\n"
        f"{artifact_note}\n\n"
        f"{DISCLAIMER}"
    )


@lru_cache(maxsize=1)
def get_runtime() -> Any:
    """Load and cache the model runtime for this Gradio process."""
    from src.deployment.inference_runtime import load_inference_runtime

    checkpoint_path = _env_path("DERMAFUSION_CKPT") or PROJECT_ROOT / "outputs/checkpoints/best.ckpt"
    return load_inference_runtime(
        project_root=PROJECT_ROOT,
        bundle_path=_env_path("DERMAFUSION_BUNDLE"),
        checkpoint_path=checkpoint_path,
        device_override=os.environ.get("DERMAFUSION_DEVICE"),
    )


def warm_runtime() -> str:
    """Load the runtime and report its status without running an image prediction."""
    try:
        runtime = get_runtime()
    except Exception as exc:
        return (
            "### Model status\n"
            "Runtime failed to load.\n\n"
            f"`{type(exc).__name__}: {exc}`\n\n"
            f"{DISCLAIMER}"
        )

    return (
        "### Model status\n"
        f"{_model_source_note(runtime)}\n\n"
        f"{runtime.checkpoint_note}\n\n"
        f"{DISCLAIMER}"
    )


def _heatmap_to_overlay(image_np: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heat = np.clip(heatmap, 0.0, 1.0)
    if heat.shape != image_np.shape[:2]:
        heat_img = Image.fromarray((heat * 255).astype(np.uint8)).resize(
            (image_np.shape[1], image_np.shape[0]),
            Image.Resampling.BILINEAR,
        )
        heat = np.asarray(heat_img).astype(np.float32) / 255.0

    overlay = image_np.astype(np.float32) / 255.0
    overlay[..., 0] = np.clip(overlay[..., 0] + 0.52 * heat, 0.0, 1.0)
    overlay[..., 1] = np.clip(overlay[..., 1] - 0.18 * heat, 0.0, 1.0)
    overlay[..., 2] = np.clip(overlay[..., 2] - 0.22 * heat, 0.0, 1.0)
    return (overlay * 255.0).astype(np.uint8)


def _preprocessed_tensor_to_image(
    image_batch: Any,
    mean: list[float],
    std: list[float],
) -> np.ndarray:
    """Convert normalized model input tensor into an RGB display image."""
    import torch

    tensor = image_batch.squeeze(0).detach().cpu().float()
    mean_t = torch.tensor(mean, dtype=tensor.dtype).view(3, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype).view(3, 1, 1)
    denorm = torch.clamp(tensor * std_t + mean_t, 0.0, 1.0)
    image_np = denorm.permute(1, 2, 0).numpy()
    return (image_np * 255.0).astype(np.uint8)


def _safe_overlay(
    runtime: Any,
    image_batch: Any,
    preprocessed_image_np: np.ndarray,
    target_class: int,
    metadata_batch: Any | None,
) -> tuple[np.ndarray, str | None]:
    from src.evaluation.interpretability import attention_rollout, generate_gradcam

    try:
        if runtime.use_metadata:
            heatmap = attention_rollout(runtime.model, image_batch.squeeze(0).cpu())
        else:
            runtime.model.zero_grad(set_to_none=True)
            heatmap = generate_gradcam(
                runtime.model,
                image_batch.squeeze(0).to(runtime.device),
                target_class=target_class,
                metadata=metadata_batch.to(runtime.device) if metadata_batch is not None else None,
            )
        return _heatmap_to_overlay(preprocessed_image_np, heatmap), None
    except Exception as exc:
        note = (
            "Interpretability overlay unavailable for this run. "
            f"{type(exc).__name__}: {exc}"
        )
        return preprocessed_image_np, note


def _summary_markdown(
    runtime: Any,
    prob_dict: dict[str, float],
    probs: np.ndarray,
    overlay_note: str | None,
) -> str:
    ranked = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_label, top_prob = ranked[0] if ranked else ("No class", 0.0)
    second_label, second_prob = ranked[1] if len(ranked) > 1 else ("No secondary class", 0.0)
    mel_prob, high_mel_risk = runtime.melanoma_decision(probs)
    max_prob = float(np.max(probs)) if len(probs) else 0.0

    confidence = "High" if max_prob >= 0.75 else "Moderate" if max_prob >= 0.45 else "Low"
    triage = "Elevated melanoma screen" if high_mel_risk else "Below melanoma threshold"
    triage_detail = (
        "Escalate to clinical review. Do not use this demo to rule in disease."
        if high_mel_risk
        else "Still review clinically if the lesion is changing, bleeding, painful, or otherwise concerning."
    )
    overlay_line = f"\n\n**Overlay note:** {overlay_note}" if overlay_note else ""

    return (
        "### Analysis summary\n"
        f"**Top prediction:** {top_label} at {top_prob:.1%}\n\n"
        f"**Second:** {second_label} at {second_prob:.1%}\n\n"
        f"**Confidence band:** {confidence}\n\n"
        f"**Melanoma probability:** {mel_prob:.1%} "
        f"(threshold {runtime.mel_threshold:.0%})\n\n"
        f"**Triage:** {triage}. {triage_detail}"
        f"{overlay_line}\n\n"
        f"{DISCLAIMER}"
    )


def _failure_artifacts(message: str, class_names: list[str] | None = None) -> PredictionArtifacts:
    return PredictionArtifacts(
        probabilities=_empty_probabilities(class_names),
        overlay=_empty_overlay(),
        summary_markdown=(
            "### Analysis unavailable\n"
            f"{message}\n\n"
            f"{DISCLAIMER}"
        ),
        model_status_markdown=_initial_status_markdown(),
    )


def predict(image: Image.Image | None, age: float, sex: str, localization: str) -> tuple[Any, ...]:
    """Run inference and return values for the Gradio UI."""
    if image is None:
        result = _failure_artifacts(
            "Upload a dermatoscopic image to begin. Metadata can stay as unknown if unavailable."
        )
        return result.probabilities, result.overlay, result.summary_markdown, result.model_status_markdown

    try:
        runtime = get_runtime()
    except Exception as exc:
        result = _failure_artifacts(
            "The runtime could not load the model bundle or checkpoint.\n\n"
            f"`{type(exc).__name__}: {exc}`"
        )
        return result.probabilities, result.overlay, result.summary_markdown, result.model_status_markdown

    try:
        image_rgb = image.convert("RGB")
        image_batch = runtime.preprocess_image(image_rgb)
        preprocessed_image_np = _preprocessed_tensor_to_image(
            image_batch,
            mean=runtime.mean,
            std=runtime.std,
        )

        metadata_batch = None
        if runtime.use_metadata:
            metadata_batch = runtime.encode_metadata(age, sex, localization).unsqueeze(0)

        probs = runtime.predict_proba(image_batch=image_batch, metadata_batch=metadata_batch)
        prob_dict = _normalise_probabilities(runtime, probs)
        target_class = int(np.argmax(probs)) if len(probs) else 0
        overlay, overlay_note = _safe_overlay(
            runtime=runtime,
            image_batch=image_batch,
            preprocessed_image_np=preprocessed_image_np,
            target_class=target_class,
            metadata_batch=metadata_batch,
        )

        summary = _summary_markdown(
            runtime=runtime,
            prob_dict=prob_dict,
            probs=probs,
            overlay_note=overlay_note,
        )
        status = (
            "### Model status\n"
            f"{_model_source_note(runtime)}\n\n"
            f"{runtime.checkpoint_note}\n\n"
            f"{DISCLAIMER}"
        )
        return prob_dict, overlay, summary, status
    except Exception as exc:
        result = _failure_artifacts(
            "Analysis failed for this input. Try another dermatoscopic image and verify the artifact path.\n\n"
            f"`{type(exc).__name__}: {exc}`",
            class_names=runtime.class_names,
        )
        return result.probabilities, result.overlay, result.summary_markdown, result.model_status_markdown


def _local_examples() -> list[list[Any]]:
    candidates = [
        PROJECT_ROOT
        / "DermFusion/DermFusion/Assets.xcassets/raw.imageset/raw_ISIC_0024306.jpg",
        PROJECT_ROOT
        / "DermFusion/DermFusion/Assets.xcassets/preprocessed.imageset/preprocessed_ISIC_0024306.jpg",
    ]
    examples: list[list[Any]] = []
    for path in candidates:
        if path.exists():
            examples.append([str(path), DEFAULT_AGE, "unknown", "unknown"])
    return examples


CUSTOM_CSS = """
:root {
  --df-bg: #07130f;
  --df-panel: rgba(15, 29, 24, 0.86);
  --df-panel-strong: rgba(20, 42, 35, 0.94);
  --df-border: rgba(142, 219, 194, 0.22);
  --df-accent: #7ce0c3;
  --df-accent-2: #8fb7ff;
  --df-text-soft: #b8c9c2;
}

.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  color: #eef8f4;
}

body,
.gradio-container {
  background:
    radial-gradient(circle at top left, rgba(124, 224, 195, 0.13), transparent 30rem),
    linear-gradient(145deg, #06100d 0%, #0b1714 46%, #10151c 100%) !important;
}

#df-hero {
  padding: 28px 0 8px;
}

#df-hero h1 {
  margin-bottom: 8px;
  font-size: clamp(2rem, 5vw, 4.2rem);
  line-height: 0.98;
  letter-spacing: 0;
}

#df-hero p {
  max-width: 760px;
  color: var(--df-text-soft);
  font-size: 1rem;
}

#df-alert {
  border: 1px solid rgba(255, 209, 102, 0.35);
  background: rgba(255, 209, 102, 0.08);
  border-radius: 14px;
  padding: 12px 14px;
}

#df-input-panel,
#df-result-panel {
  border: 1px solid var(--df-border);
  background: var(--df-panel);
  border-radius: 18px;
  padding: 16px;
  box-shadow: 0 18px 54px rgba(0, 0, 0, 0.28);
}

#df-input-panel .wrap,
#df-result-panel .wrap {
  background: transparent;
}

#df-analyze {
  border: 0;
  background: linear-gradient(135deg, var(--df-accent), var(--df-accent-2));
  color: #06100d;
  font-weight: 800;
}

#df-secondary {
  border: 1px solid var(--df-border);
  background: var(--df-panel-strong);
}

#df-status,
#df-summary {
  color: #edf7f3;
}

#df-status code,
#df-summary code {
  white-space: normal;
}

@media (max-width: 720px) {
  #df-hero {
    padding-top: 18px;
  }

  #df-input-panel,
  #df-result-panel {
    padding: 12px;
    border-radius: 14px;
  }
}
"""


def build_demo() -> gr.Blocks:
    """Create the complete Gradio UI."""
    with gr.Blocks(
        title="DermaFusion Research Demo",
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(primary_hue="teal", secondary_hue="blue", neutral_hue="slate"),
    ) as blocks:
        with gr.Column(elem_id="df-hero"):
            gr.Markdown(
                "# DermaFusion\n"
                "Multiclass skin lesion research demo with model probabilities, melanoma thresholding, "
                "and a visual saliency overlay."
            )
            gr.Markdown(f"**Clinical boundary:** {DISCLAIMER}", elem_id="df-alert")

        with gr.Row(equal_height=False):
            with gr.Column(scale=5, elem_id="df-input-panel"):
                gr.Markdown("### Input")
                image_input = gr.Image(
                    type="pil",
                    label="Dermatoscopic image",
                    sources=["upload", "webcam", "clipboard"],
                    height=360,
                )
                with gr.Accordion("Optional patient metadata", open=True):
                    age_input = gr.Slider(
                        minimum=0,
                        maximum=100,
                        value=DEFAULT_AGE,
                        step=1,
                        label="Age",
                    )
                    with gr.Row():
                        sex_input = gr.Dropdown(SEX_VALUES, value="unknown", label="Sex")
                        location_input = gr.Dropdown(
                            LOCALIZATION_VALUES,
                            value="unknown",
                            label="Lesion location",
                        )

                with gr.Row():
                    analyze_button = gr.Button("Analyze lesion", variant="primary", elem_id="df-analyze")
                    warm_button = gr.Button("Warm up model", elem_id="df-secondary")
                    clear_button = gr.ClearButton(value="Clear")

                examples = _local_examples()
                if examples:
                    gr.Examples(
                        examples=examples,
                        inputs=[image_input, age_input, sex_input, location_input],
                        label="Local sample images",
                    )

            with gr.Column(scale=6, elem_id="df-result-panel"):
                gr.Markdown("### Results")
                summary_output = gr.Markdown(
                    "Upload an image, confirm metadata if available, then run analysis.",
                    elem_id="df-summary",
                )
                probabilities_output = gr.Label(
                    num_top_classes=len(DEFAULT_CLASS_ORDER),
                    label="Diagnosis probabilities",
                    value=_empty_probabilities(),
                )
                overlay_output = gr.Image(
                    label="Saliency overlay on preprocessed image",
                    value=_empty_overlay(),
                    height=360,
                )
                status_output = gr.Markdown(_initial_status_markdown(), elem_id="df-status")

        analyze_button.click(
            fn=predict,
            inputs=[image_input, age_input, sex_input, location_input],
            outputs=[probabilities_output, overlay_output, summary_output, status_output],
        )
        warm_button.click(fn=warm_runtime, inputs=None, outputs=status_output)
        clear_button.add(
            [
                image_input,
                probabilities_output,
                overlay_output,
                summary_output,
                status_output,
            ]
        )

    return blocks.queue()


demo = build_demo()


def launch_demo() -> None:
    """Launch the local Gradio app."""
    _extend_gradio_start_timeout()
    _relax_gradio_local_url_check()
    host = os.environ.get("DERMAFUSION_GRADIO_HOST", "127.0.0.1")
    launch_kwargs: dict[str, Any] = {
        "server_name": host,
        "show_error": True,
        "share": _env_flag("DERMAFUSION_GRADIO_SHARE", False),
    }
    explicit_port = (
        os.environ.get("DERMAFUSION_GRADIO_PORT", "").strip()
        or os.environ.get("GRADIO_SERVER_PORT", "").strip()
    )
    launch_kwargs["server_port"] = int(explicit_port) if explicit_port else _available_port(host)

    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_demo()
