"""Evaluation entrypoint for trained models."""

from __future__ import annotations

import json
import os
from pathlib import Path
import sys
from typing import Any

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HAM10000Dataset
from src.data.preprocessing import encode_metadata
from src.data.transforms import get_val_transforms
from src.evaluation.calibration import expected_calibration_error, plot_reliability_diagram
from src.evaluation.metrics import MetricCalculator
from src.evaluation.statistical import bootstrap_confidence_intervals, compute_all_cis
from src.models.model_factory import build_model
from src.utils.config import validate_config
from src.utils.logging import setup_logger
from src.utils.reproducibility import get_device, seed_everything

CLASS_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}
GT_CLASS_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def _resolve_path(path_value: str) -> Path:
    """Resolve config path to absolute project path."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_aux_metadata(metadata_csv: Path) -> pd.DataFrame | None:
    """Load optional metadata table with age/sex/localization."""
    if not metadata_csv.exists():
        return None
    df = pd.read_csv(metadata_csv)
    if "image_id" not in df.columns:
        return None
    return df


def _build_split_df(
    split: str,
    metadata_dir: Path,
    aux_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build split dataframe from official ISIC ground-truth files."""
    split_to_csv = {
        "train": "ISIC2018_Task3_Training_GroundTruth.csv",
        "val": "ISIC2018_Task3_Validation_GroundTruth.csv",
        "test": "ISIC2018_Task3_Test_GroundTruth.csv",
    }
    gt_csv = metadata_dir / split_to_csv[split]
    gt_df = pd.read_csv(gt_csv).rename(columns={"image": "image_id"})
    for col in GT_CLASS_COLS:
        gt_df[col] = pd.to_numeric(gt_df[col], errors="coerce").fillna(0.0)
    gt_df["dx"] = gt_df[GT_CLASS_COLS].idxmax(axis=1).str.lower()
    gt_df["label"] = gt_df["dx"].map(CLASS_TO_IDX).astype(int)
    gt_df["dx_type"] = "official_split"
    gt_df["age"] = pd.NA
    gt_df["sex"] = "unknown"
    gt_df["localization"] = "unknown"

    if split == "train":
        lesion_csv = metadata_dir / "ISIC2018_Task3_Training_LesionGroupings.csv"
        lesion_df = pd.read_csv(lesion_csv).rename(columns={"image": "image_id"})
        gt_df = gt_df.merge(lesion_df[["image_id", "lesion_id"]], on="image_id", how="left")
        gt_df["lesion_id"] = gt_df["lesion_id"].fillna(gt_df["image_id"])
    else:
        gt_df["lesion_id"] = gt_df["image_id"]

    if aux_metadata is not None:
        gt_df = gt_df.merge(
            aux_metadata[["image_id", "age", "sex", "localization", "dx_type"]].drop_duplicates(
                "image_id"
            ),
            on="image_id",
            how="left",
            suffixes=("", "_aux"),
        )
        gt_df["age"] = gt_df["age_aux"].where(gt_df["age_aux"].notna(), gt_df["age"])
        gt_df["sex"] = gt_df["sex_aux"].where(gt_df["sex_aux"].notna(), gt_df["sex"])
        gt_df["localization"] = gt_df["localization_aux"].where(
            gt_df["localization_aux"].notna(), gt_df["localization"]
        )
        gt_df["dx_type"] = gt_df["dx_type_aux"].where(gt_df["dx_type_aux"].notna(), gt_df["dx_type"])
        gt_df = gt_df.drop(
            columns=["age_aux", "sex_aux", "localization_aux", "dx_type_aux"], errors="ignore"
        )

    return gt_df


def _split_summary(df: pd.DataFrame) -> dict[str, Any]:
    """Create sample and lesion-level summary for one split."""
    class_counts = {
        class_name: int((df["label"] == idx).sum()) for idx, class_name in enumerate(CLASS_ORDER)
    }
    return {
        "samples": int(len(df)),
        "unique_lesions": int(df["lesion_id"].nunique()),
        "class_counts": class_counts,
    }


def _build_data(cfg: DictConfig) -> tuple[dict[str, DataLoader], dict[str, dict[str, Any]]]:
    """Build train/validation/test dataloaders plus split summaries."""
    image_size = int(OmegaConf.select(cfg, "data.data.image_size", default=380))
    batch_size = int(OmegaConf.select(cfg, "training.training.batch_size", default=32))
    num_workers = int(OmegaConf.select(cfg, "data.data.num_workers", default=4))
    pin_memory = bool(OmegaConf.select(cfg, "data.data.pin_memory", default=True))
    validate_images_on_init = bool(OmegaConf.select(cfg, "data.data.validate_images_on_init", default=True))
    image_root = _resolve_path(str(cfg.data.data.image_dir))
    metadata_csv = _resolve_path(str(cfg.data.data.metadata_csv))
    metadata_dir = metadata_csv.parent
    if not metadata_csv.exists():
        metadata_csv = metadata_dir / "metadata_merged.csv"
    aux_metadata = _load_aux_metadata(metadata_csv)

    train_df = _build_split_df(split="train", metadata_dir=metadata_dir, aux_metadata=aux_metadata)
    val_df = _build_split_df(split="val", metadata_dir=metadata_dir, aux_metadata=aux_metadata)
    test_df = _build_split_df(split="test", metadata_dir=metadata_dir, aux_metadata=aux_metadata)

    train_df, encoding_stats = encode_metadata(train_df)
    val_df, _ = encode_metadata(val_df, stats=encoding_stats)
    test_df, _ = encode_metadata(test_df, stats=encoding_stats)

    train_dataset = HAM10000Dataset(
        df=train_df,
        image_dir=image_root / "train",
        transform=get_val_transforms(image_size=image_size),
        validate_images_on_init=validate_images_on_init,
    )
    val_dataset = HAM10000Dataset(
        df=val_df,
        image_dir=image_root / "val",
        transform=get_val_transforms(image_size=image_size),
        validate_images_on_init=validate_images_on_init,
    )
    test_dataset = HAM10000Dataset(
        df=test_df,
        image_dir=image_root / "test",
        transform=get_val_transforms(image_size=image_size),
        validate_images_on_init=validate_images_on_init,
    )
    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
    summary = {
        "train": _split_summary(train_df),
        "val": _split_summary(val_df),
        "test": _split_summary(test_df),
    }
    return loaders, summary


@torch.no_grad()
def _run_inference(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect true labels, predicted labels, and class probabilities for one split."""
    model.eval()
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []
    for batch in dataloader:
        batch_for_model = {
            key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()
        }
        logits = model(batch_for_model)
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        all_true.append(batch["label"].cpu().numpy().astype(int))
        all_pred.append(preds.cpu().numpy().astype(int))
        all_prob.append(probs.cpu().numpy())
    return np.concatenate(all_true), np.concatenate(all_pred), np.concatenate(all_prob)


def _resolve_checkpoint_path() -> Path:
    """Resolve checkpoint path from env or default output location."""
    env_value = os.environ.get("DERMAFUSION_CKPT", "").strip()
    if env_value:
        env_ckpt = Path(env_value).expanduser()
        if env_ckpt.exists():
            return env_ckpt
    default_path = PROJECT_ROOT / "outputs" / "checkpoints" / "best.ckpt"
    if default_path.exists():
        return default_path
    fallback_dir = PROJECT_ROOT / "outputs" / "checkpoints"
    for pattern in ("*.ckpt", "*.pt", "*.pth"):
        matches = sorted(fallback_dir.glob(pattern))
        if matches:
            return matches[0]
    return default_path


def _to_float(value: Any) -> float:
    """Convert scalar-like values to float."""
    return float(value) if value is not None else float("nan")


def _write_markdown_report(
    report_path: Path,
    checkpoint_path: Path,
    data_summary: dict[str, dict[str, Any]],
    metric_summary: dict[str, dict[str, Any]],
) -> None:
    """Write human-readable markdown report for train/val/test and evaluation metrics."""
    lines: list[str] = []
    lines.append("# Model Training and Evaluation Report")
    lines.append("")
    lines.append(f"- Checkpoint: `{checkpoint_path}`")
    lines.append("")
    lines.append("## Data Summary")
    lines.append("")
    lines.append("| Split | Samples | Unique Lesions |")
    lines.append("|---|---:|---:|")
    for split in ["train", "val", "test"]:
        summary = data_summary[split]
        lines.append(f"| {split} | {summary['samples']} | {summary['unique_lesions']} |")
    lines.append("")
    lines.append("### Class Distribution")
    lines.append("")
    lines.append("| Split | mel | nv | bcc | akiec | bkl | df | vasc |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split in ["train", "val", "test"]:
        counts = data_summary[split]["class_counts"]
        lines.append(
            "| "
            f"{split} | {counts['mel']} | {counts['nv']} | {counts['bcc']} | "
            f"{counts['akiec']} | {counts['bkl']} | {counts['df']} | {counts['vasc']} |"
        )
    lines.append("")

    lines.append("## Evaluation Metrics")
    lines.append("")
    lines.append("| Split | Balanced Accuracy | Macro F1 | Weighted F1 | Cohen's Kappa | ECE |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for split in ["train", "val", "test"]:
        metrics = metric_summary[split]
        lines.append(
            f"| {split} | {metrics['balanced_accuracy']:.4f} | {metrics['macro_f1']:.4f} | "
            f"{metrics['weighted_f1']:.4f} | {metrics['cohens_kappa']:.4f} | {metrics['ece']:.4f} |"
        )
    lines.append("")

    for split in ["train", "val", "test"]:
        lines.append(f"### Per-Class Metrics ({split})")
        lines.append("")
        lines.append("| Class | Sensitivity | Specificity | AUC-ROC | AUC-PR | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        per_class = metric_summary[split]["per_class"]
        for class_name in CLASS_ORDER:
            lines.append(
                f"| {class_name} | "
                f"{_to_float(per_class['sensitivity'].get(class_name)):.4f} | "
                f"{_to_float(per_class['specificity'].get(class_name)):.4f} | "
                f"{_to_float(per_class['auc_roc'].get(class_name)):.4f} | "
                f"{_to_float(per_class['auc_pr'].get(class_name)):.4f} | "
                f"{_to_float(per_class['f1'].get(class_name)):.4f} |"
            )
        lines.append("")
        lines.append(f"- Reliability diagram: `outputs/reports/reliability_{split}.png`")
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Run evaluation using trained checkpoint and write JSON/Markdown reports."""
    validate_config(cfg)
    seed_everything(int(cfg.seed))
    logger = setup_logger("evaluate", log_file=PROJECT_ROOT / "outputs/logs/evaluate.log")
    device = get_device()

    model = build_model(cfg).to(device)
    checkpoint_path = _resolve_checkpoint_path()
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location=device)
        state = payload.get("model_state", payload)
        model.load_state_dict(state, strict=False)
        logger.info("Loaded checkpoint: %s", checkpoint_path)
    else:
        logger.warning("Checkpoint not found at %s; using current model weights.", checkpoint_path)

    loaders, data_summary = _build_data(cfg)
    metric_summary: dict[str, dict[str, Any]] = {}

    for split_name in ["train", "val", "test"]:
        y_true, y_pred, y_prob = _run_inference(model=model, dataloader=loaders[split_name], device=device)

        calculator = MetricCalculator(num_classes=7, class_names=CLASS_ORDER)
        calculator.update(predictions=y_pred, labels=y_true, probabilities=y_prob)
        metrics = calculator.compute()
        metrics["ece"] = expected_calibration_error(y_prob, y_true, n_bins=10)

        plot_reliability_diagram(
            y_prob,
            y_true,
            save_path=PROJECT_ROOT / "outputs" / "reports" / f"reliability_{split_name}.png",
        )
        metrics["ci_accuracy"] = compute_all_cis({"accuracy": (y_true, y_pred)})["accuracy"]
        metrics["ci_balanced_accuracy"] = bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred,
            metric_fn=lambda yt, yp: float(balanced_accuracy_score(yt, yp)),
        )
        metrics["ci_macro_f1"] = bootstrap_confidence_intervals(
            y_true=y_true,
            y_pred=y_pred,
            metric_fn=lambda yt, yp: float(f1_score(yt, yp, average="macro")),
        )
        metric_summary[split_name] = metrics

    report_payload = {
        "checkpoint": str(checkpoint_path),
        "data_summary": data_summary,
        "metrics": metric_summary,
    }
    report_dir = PROJECT_ROOT / "outputs" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "evaluation_summary.json").write_text(
        json.dumps(report_payload, indent=2, default=lambda x: x.tolist() if hasattr(x, "tolist") else x),
        encoding="utf-8",
    )
    markdown_path = report_dir / "model_training_evaluation_report.md"
    _write_markdown_report(
        report_path=markdown_path,
        checkpoint_path=checkpoint_path,
        data_summary=data_summary,
        metric_summary=metric_summary,
    )
    logger.info("Wrote evaluation markdown report to %s", markdown_path)


if __name__ == "__main__":
    main()
