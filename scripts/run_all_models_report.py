"""Run model variants and generate a consolidated performance report."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_VARIANTS: list[dict[str, Any]] = [
    {
        "name": "efficientnet_image_only",
        "overrides": [
            "experiment=baseline",
            "model.model.backbone=efficientnet_b4",
            "model.model.use_metadata=false",
        ],
    },
    {
        "name": "efficientnet_late_fusion",
        "overrides": [
            "experiment=multimodal_late",
            "model.model.backbone=efficientnet_b4",
            "model.model.use_metadata=true",
            "model.model.fusion=late",
        ],
    },
    {
        "name": "efficientnet_film",
        "overrides": [
            "experiment=multimodal_film",
            "model.model.backbone=efficientnet_b4",
            "model.model.use_metadata=true",
            "model.model.fusion=film",
        ],
    },
    {
        "name": "efficientnet_cross_attention",
        "overrides": [
            "model.model.backbone=efficientnet_b4",
            "model.model.use_metadata=true",
            "model.model.fusion=cross_attention",
        ],
    },
    {
        "name": "swin_image_only",
        "overrides": [
            "experiment=baseline",
            "model.model.backbone=swin_tiny",
            "model.model.use_metadata=false",
        ],
    },
    {
        "name": "swin_late_fusion",
        "overrides": [
            "experiment=multimodal_late",
            "model.model.backbone=swin_tiny",
            "model.model.use_metadata=true",
            "model.model.fusion=late",
        ],
    },
    {
        "name": "convnext_image_only",
        "overrides": [
            "experiment=baseline",
            "model.model.backbone=convnext_v2",
            "model.model.use_metadata=false",
        ],
    },
    {
        "name": "convnext_late_fusion",
        "overrides": [
            "experiment=multimodal_late",
            "model.model.backbone=convnext_v2",
            "model.model.use_metadata=true",
            "model.model.fusion=late",
        ],
    },
]


def _run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    """Run command and raise on failure."""
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def _read_eval_json(path: Path) -> dict[str, Any]:
    """Read evaluation summary json."""
    return json.loads(path.read_text(encoding="utf-8"))


def _write_report(results: list[dict[str, Any]], output_path: Path) -> None:
    """Write consolidated markdown report for all model variants."""
    lines: list[str] = []
    lines.append("# All Models Performance Report")
    lines.append("")
    lines.append("| Model | Train Balanced Acc | Test Balanced Acc | Train Macro F1 | Test Macro F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in results:
        lines.append(
            f"| {row['name']} | {row['train_bal_acc']:.4f} | {row['test_bal_acc']:.4f} | "
            f"{row['train_macro_f1']:.4f} | {row['test_macro_f1']:.4f} |"
        )
    lines.append("")
    lines.append("Detailed per-run outputs:")
    lines.append("- `outputs/reports/evaluation_summary_<model>.json`")
    lines.append("- `outputs/reports/model_training_evaluation_report_<model>.md`")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Train/evaluate all model variants.")
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    """Run all variants and generate consolidated report."""
    args = parse_args()
    reports_dir = PROJECT_ROOT / "outputs" / "reports"
    ckpt_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    reports_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for variant in MODEL_VARIANTS:
        name = str(variant["name"])
        train_cmd = [
            args.python,
            "scripts/train.py",
            f"experiment_name={name}",
            f"training.training.epochs={args.epochs}",
            f"training.training.batch_size={args.batch_size}",
            *list(variant["overrides"]),
        ]
        _run(train_cmd)

        best_ckpt = ckpt_dir / "best.ckpt"
        named_ckpt = ckpt_dir / f"{name}.ckpt"
        if best_ckpt.exists():
            shutil.copy2(best_ckpt, named_ckpt)

        env = os.environ.copy()
        env["DERMAFUSION_CKPT"] = str(named_ckpt)
        eval_cmd = [args.python, "scripts/evaluate.py"]
        _run(eval_cmd, env=env)

        eval_json = reports_dir / "evaluation_summary.json"
        eval_md = reports_dir / "model_training_evaluation_report.md"
        model_eval_json = reports_dir / f"evaluation_summary_{name}.json"
        model_eval_md = reports_dir / f"model_training_evaluation_report_{name}.md"
        shutil.copy2(eval_json, model_eval_json)
        shutil.copy2(eval_md, model_eval_md)

        payload = _read_eval_json(model_eval_json)
        metrics = payload["metrics"]
        results.append(
            {
                "name": name,
                "train_bal_acc": float(metrics["train"]["balanced_accuracy"]),
                "test_bal_acc": float(metrics["test"]["balanced_accuracy"]),
                "train_macro_f1": float(metrics["train"]["macro_f1"]),
                "test_macro_f1": float(metrics["test"]["macro_f1"]),
            }
        )

    (reports_dir / "all_models_performance.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    _write_report(results=results, output_path=reports_dir / "all_models_performance.md")


if __name__ == "__main__":
    main()
