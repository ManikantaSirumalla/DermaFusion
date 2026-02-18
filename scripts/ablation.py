"""Run ablation experiments and produce a comparison summary."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logging import setup_logger


@dataclass(frozen=True)
class AblationExperiment:
    """Container for one ablation experiment definition."""

    name: str
    overrides: dict[str, Any]


ABLATION_EXPERIMENTS: list[AblationExperiment] = [
    AblationExperiment(
        "01_efficientnet_image_only",
        {"model.model.use_metadata": False, "model.model.backbone": "efficientnet_b4"},
    ),
    AblationExperiment(
        "02_efficientnet_late_fusion",
        {"model.model.use_metadata": True, "model.model.fusion": "late"},
    ),
    AblationExperiment(
        "03_efficientnet_film",
        {"model.model.use_metadata": True, "model.model.fusion": "film"},
    ),
    AblationExperiment(
        "04_efficientnet_cross_attention",
        {"model.model.use_metadata": True, "model.model.fusion": "cross_attention"},
    ),
    AblationExperiment(
        "05_swin_image_only",
        {"model.model.backbone": "swin_tiny", "model.model.use_metadata": False},
    ),
    AblationExperiment(
        "06_swin_late_fusion",
        {"model.model.backbone": "swin_tiny", "model.model.fusion": "late", "model.model.use_metadata": True},
    ),
    AblationExperiment(
        "07_convnext_image_only",
        {"model.model.backbone": "convnext_v2", "model.model.use_metadata": False},
    ),
    AblationExperiment(
        "08_convnext_late_fusion",
        {"model.model.backbone": "convnext_v2", "model.model.fusion": "late", "model.model.use_metadata": True},
    ),
    AblationExperiment("09_no_augmentation", {"data.data.augmentation": "none"}),
    AblationExperiment("10_no_class_weighting", {"training.training.loss.alpha": "none"}),
    AblationExperiment("11_frozen_backbone", {"training.training.freeze_backbone": True}),
    AblationExperiment("12_random_init", {"model.model.pretrained": False}),
    AblationExperiment("13_ensemble_top3", {}),
]


def _to_hydra_override(key: str, value: Any) -> str:
    """Convert key-value pair into hydra override string."""
    if isinstance(value, bool):
        return f"{key}={'true' if value else 'false'}"
    return f"{key}={value}"


def run_ablation_suite(
    python_executable: str,
    dry_run: bool,
    output_csv: Path,
) -> list[dict[str, str]]:
    """Run all ablation experiments via train/evaluate scripts."""
    logger = setup_logger("ablation", log_file=PROJECT_ROOT / "outputs/logs/ablation.log")
    results: list[dict[str, str]] = []

    for exp in ABLATION_EXPERIMENTS:
        logger.info("Starting ablation experiment: %s", exp.name)
        overrides = [_to_hydra_override(k, v) for k, v in exp.overrides.items()]
        train_cmd = [python_executable, "scripts/train.py", *overrides]
        eval_cmd = [python_executable, "scripts/evaluate.py", *overrides]

        status = "success"
        detail = "completed"
        if dry_run:
            logger.info("DRY RUN train cmd: %s", " ".join(train_cmd))
            logger.info("DRY RUN eval cmd: %s", " ".join(eval_cmd))
            status = "dry_run"
            detail = "not executed"
        else:
            train_proc = subprocess.run(train_cmd, cwd=PROJECT_ROOT, check=False, text=True)
            if train_proc.returncode != 0:
                status = "failed"
                detail = f"train_failed_{train_proc.returncode}"
            else:
                eval_proc = subprocess.run(eval_cmd, cwd=PROJECT_ROOT, check=False, text=True)
                if eval_proc.returncode != 0:
                    status = "failed"
                    detail = f"eval_failed_{eval_proc.returncode}"

        results.append({"experiment": exp.name, "status": status, "detail": detail})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["experiment", "status", "detail"])
        writer.writeheader()
        writer.writerows(results)
    logger.info("Ablation summary written to: %s", output_csv)
    return results


def parse_args() -> argparse.Namespace:
    """Parse ablation runner arguments."""
    parser = argparse.ArgumentParser(description="Run ablation experiment suite.")
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python executable used to launch train/evaluate scripts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and generate summary without executing runs.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "ablation" / "ablation_results.csv",
        help="Path to write ablation summary CSV.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    run_ablation_suite(
        python_executable=args.python,
        dry_run=args.dry_run,
        output_csv=args.output_csv,
    )


if __name__ == "__main__":
    main()
