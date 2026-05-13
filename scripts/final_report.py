"""Generate final markdown report + model card from evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build final DermaFusion report artifacts.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root containing outputs/.",
    )
    parser.add_argument(
        "--mel-threshold",
        type=float,
        default=0.30,
        help="Deployment threshold used for melanoma risk flagging.",
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt(x: Any) -> str:
    try:
        return f"{float(x):.4f}"
    except Exception:
        return "n/a"


def main() -> None:
    args = parse_args()
    root = args.project_root
    outputs = root / "outputs"
    reports_dir = outputs / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    eval_summary = _read_json(reports_dir / "evaluation_summary.json")
    final_metrics = _read_json(outputs / "final_metrics_v2.json")

    metrics = eval_summary.get("metrics", {})
    data_summary = eval_summary.get("data_summary", {})
    ckpt = eval_summary.get("checkpoint", str(outputs / "checkpoints/best.ckpt"))

    lines: list[str] = []
    lines.append("# DermaFusion Final Report")
    lines.append("")
    lines.append("## 1. Model And Artifacts")
    lines.append(f"- Checkpoint: `{ckpt}`")
    lines.append(f"- Bundle: `{outputs / 'export/dermafusion_bundle.pt'}`")
    lines.append(f"- MEL threshold: `{args.mel_threshold:.2f}`")
    lines.append("")

    lines.append("## 2. Data Summary")
    lines.append("| Split | Samples | Unique Lesions |")
    lines.append("|---|---:|---:|")
    for split in ("train", "val", "test"):
        s = data_summary.get(split, {})
        lines.append(f"| {split} | {s.get('samples', 'n/a')} | {s.get('unique_lesions', 'n/a')} |")
    lines.append("")

    lines.append("## 3. Metrics (Argmax)")
    lines.append("| Split | Balanced Accuracy | Macro F1 | Weighted F1 | Cohen Kappa | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in ("train", "val", "test"):
        s = metrics.get(split, {})
        lines.append(
            f"| {split} | {_fmt(s.get('balanced_accuracy'))} | {_fmt(s.get('macro_f1'))} | "
            f"{_fmt(s.get('weighted_f1'))} | {_fmt(s.get('cohens_kappa'))} | {_fmt(s.get('ece'))} |"
        )
    lines.append("")

    if final_metrics:
        lines.append("## 4. Thresholded Deployment Metrics")
        lines.append("```json")
        lines.append(json.dumps(final_metrics, indent=2))
        lines.append("```")
        lines.append("")

    lines.append("## 5. Supporting Files")
    artifacts = [
        outputs / "training_dynamics_v2.png",
        outputs / "confusion_matrix_v2.png",
        outputs / "roc_curves_v2.png",
        outputs / "gradcam_v2.png",
        outputs / "fairness_report_v2.csv",
    ]
    for artifact in artifacts:
        state = "exists" if artifact.exists() else "missing"
        lines.append(f"- `{artifact}` ({state})")
    lines.append("")

    final_report = reports_dir / "final_report.md"
    final_report.write_text("\n".join(lines), encoding="utf-8")

    model_card = reports_dir / "model_card.md"
    model_card_lines = [
        "# DermaFusion Model Card",
        "",
        "## Intended Use",
        "- Research and educational skin-lesion triage support.",
        "- Not a standalone diagnostic system.",
        "",
        "## Model",
        f"- Checkpoint: `{ckpt}`",
        f"- Bundle: `{outputs / 'export/dermafusion_bundle.pt'}`",
        "- Backbone: EfficientNet-B4 (deployment bundle dependent).",
        f"- Melanoma threshold: `{args.mel_threshold:.2f}`",
        "",
        "## Inputs",
        "- RGB dermoscopic image, resized to 380x380.",
        "- Normalization mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225].",
        "- Optional metadata: age, sex, localization (when model was trained multimodal).",
        "",
        "## Outputs",
        "- Class probability distribution.",
        "- Top-k labels.",
        "- Melanoma risk flag using configured threshold.",
        "",
        "## Evaluation Snapshot",
        f"- Validation balanced accuracy: {_fmt(metrics.get('val', {}).get('balanced_accuracy'))}",
        f"- Validation macro F1: {_fmt(metrics.get('val', {}).get('macro_f1'))}",
        f"- Test balanced accuracy: {_fmt(metrics.get('test', {}).get('balanced_accuracy'))}",
        f"- Test macro F1: {_fmt(metrics.get('test', {}).get('macro_f1'))}",
        "",
        "## Limitations",
        "- Performance may vary by source domain, skin tone, device, and capture quality.",
        "- Calibration and thresholding should be validated for each deployment context.",
        "- Clinical decisions must remain with qualified professionals.",
    ]
    model_card.write_text("\n".join(model_card_lines), encoding="utf-8")

    summary_json = reports_dir / "final_report.json"
    summary_json.write_text(
        json.dumps(
            {
                "checkpoint": ckpt,
                "mel_threshold": args.mel_threshold,
                "metrics": metrics,
                "final_metrics_v2": final_metrics,
                "artifacts": [str(p) for p in artifacts],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote: {final_report}")
    print(f"Wrote: {model_card}")
    print(f"Wrote: {summary_json}")


if __name__ == "__main__":
    main()

