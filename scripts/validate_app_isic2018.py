"""Run app model on ISIC 2018 samples and produce validation report with ground truth.

Outputs:
- CSV: image_id, ground_truth_label, predicted_label, correct, top_prob, and per-class probs.
- Optional: copy sample images to an output folder for manual validation.

Usage:
  python scripts/validate_app_isic2018.py --isic-dir Datasets/ISIC2018_Task3 --out-dir outputs/validation_app
  python scripts/validate_app_isic2018.py --isic-dir Datasets/ISIC2018_Task3 --out-dir outputs/validation_app --max-per-class 5 --copy-images
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.deployment.inference_runtime import load_inference_runtime, resolve_bundle_path
from src.data.preprocessing import dull_razor, shades_of_gray

# ISIC 2018 Task 3 class columns (uppercase); model uses lowercase.
CLASS_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
SPLIT_CHOICES = ("train", "val", "test", "all")
PREPROCESS_CHOICES = ("runtime", "sog", "sog_hair")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate app model on ISIC 2018 samples with ground truth."
    )
    parser.add_argument(
        "--isic-dir",
        type=Path,
        default=PROJECT_ROOT / "Datasets" / "ISIC2018_Task3",
        help="Root of ISIC 2018 Task 3 (Training_Input, *_GroundTruth, etc.).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "validation_app",
        help="Output directory for CSV report and optional images.",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=10,
        help="Max samples per ground-truth class (stratified sample).",
    )
    parser.add_argument(
        "--max-total",
        type=int,
        default=70,
        help="Cap total number of images to run (0 = no cap).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=SPLIT_CHOICES,
        help="Which ISIC split to validate on (default: val).",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="runtime",
        choices=PREPROCESS_CHOICES,
        help="Input preprocessing before runtime transform.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy sample images into out-dir/images for manual check.",
    )
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="Bundle path (optional; auto-discovery checks V2/demo and legacy locations).",
    )
    parser.add_argument(
        "--full-split",
        action="store_true",
        help="Run the full selected split (ignores max-per-class/max-total sampling).",
    )
    parser.add_argument(
        "--mel-threshold",
        type=float,
        default=None,
        help="Optional melanoma threshold override for binary MEL triage metrics.",
    )
    return parser.parse_args()


def get_ground_truth_label(row: pd.Series) -> str:
    """From ISIC one-hot row return single label (uppercase)."""
    for c in CLASS_COLS:
        if row.get(c, 0) == 1.0:
            return c
    return "unknown"


def resolve_image_path(isic_dir: Path, image_id: str) -> Path | None:
    """Find image in Training_Input, Validation_Input, or Test_Input."""
    for sub in ("ISIC2018_Task3_Training_Input", "ISIC2018_Task3_Validation_Input", "ISIC2018_Task3_Test_Input"):
        folder = isic_dir / sub
        if not folder.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            p = folder / f"{image_id}{ext}"
            if p.exists():
                return p
    return None


def load_ground_truth(isic_dir: Path) -> pd.DataFrame:
    """Load and concatenate train/validation/test ground truth with split and image path."""
    dfs = []
    for split, name in (
        ("train", "ISIC2018_Task3_Training_GroundTruth"),
        ("val", "ISIC2018_Task3_Validation_GroundTruth"),
        ("test", "ISIC2018_Task3_Test_GroundTruth"),
    ):
        csv_flat = isic_dir / f"{name}.csv"
        csv_nested = isic_dir / name / f"{name}.csv"
        path = csv_flat if csv_flat.exists() else (csv_nested if csv_nested.exists() else None)
        if path is None:
            continue
        df = pd.read_csv(path)
        df["split"] = split
        # ISIC CSV uses 'image' column for image ID
        id_col = "image" if "image" in df.columns else "image_id"
        df["image_id"] = df[id_col].astype(str).str.strip()
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No ground truth CSVs found under {isic_dir}")
    return pd.concat(dfs, ignore_index=True)


def preprocess_pil(image: Image.Image, mode: str) -> Image.Image:
    """Apply optional dermoscopy preprocessing before runtime tensor transform."""
    if mode == "runtime":
        return image
    arr = np.asarray(image.convert("RGB"), dtype=np.uint8)
    arr = shades_of_gray(arr)
    if mode == "sog_hair":
        arr = dull_razor(arr)
    return Image.fromarray(arr.astype(np.uint8), mode="RGB")


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def main() -> None:
    args = parse_args()
    isic_dir = args.isic_dir.resolve()
    if not isic_dir.exists():
        raise FileNotFoundError(f"ISIC directory not found: {isic_dir}")

    # Load runtime (shared resolver supports V2 + legacy bundle locations).
    bundle_path = resolve_bundle_path(PROJECT_ROOT, explicit_path=args.bundle)
    if bundle_path is None:
        raise FileNotFoundError(
            "No bundle found. Set --bundle or place it in demo/dermafusion_new_V2, "
            "outputs/export, or project root."
        )

    runtime = load_inference_runtime(
        project_root=PROJECT_ROOT,
        bundle_path=bundle_path,
        checkpoint_path=PROJECT_ROOT / "outputs" / "checkpoints" / "best.ckpt",
    )
    class_names = runtime.class_names  # lowercase

    # Load GT and add ground_truth_label
    gt_all = load_ground_truth(isic_dir)
    if args.split != "all":
        gt_all = gt_all[gt_all["split"] == args.split].copy()
    gt_all["ground_truth_label"] = gt_all.apply(get_ground_truth_label, axis=1)

    # Resolve image paths and drop missing
    paths = []
    for _, row in gt_all.iterrows():
        p = resolve_image_path(isic_dir, row["image_id"])
        paths.append(p)
    gt_all["image_path"] = paths
    gt_all = gt_all[gt_all["image_path"].notna()].copy()
    gt_all["image_path"] = gt_all["image_path"].astype(str)

    # Stratified sample unless full split is requested.
    if args.full_split:
        sample_df = gt_all.reset_index(drop=True)
    else:
        sampled = []
        for label in CLASS_COLS:
            subset = gt_all[gt_all["ground_truth_label"] == label]
            n = len(subset) if args.max_per_class <= 0 else min(len(subset), args.max_per_class)
            if n > 0:
                sampled.append(subset.sample(n=n, random_state=42))
        if not sampled:
            raise ValueError("No samples found; check ground truth and image paths.")
        sample_df = pd.concat(sampled, ignore_index=True)
        if args.max_total > 0 and len(sample_df) > args.max_total:
            sample_df = sample_df.sample(n=args.max_total, random_state=43).reset_index(drop=True)

    # Run inference (no metadata for simplicity; app uses optional metadata)
    class_to_idx_upper = {name.upper(): idx for idx, name in enumerate(class_names)}
    results = []
    out_images = args.out_dir / "images" if args.copy_images else None
    if out_images:
        out_images.mkdir(parents=True, exist_ok=True)

    for _, row in sample_df.iterrows():
        image_path = Path(row["image_path"])
        image_id = row["image_id"]
        gt_label = row["ground_truth_label"]

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            results.append({
                "image_id": image_id,
                "ground_truth_label": gt_label,
                "predicted_label": "error",
                "correct": False,
                "error": str(e),
            })
            continue

        image_for_model = preprocess_pil(image, args.preprocess)
        image_batch = runtime.preprocess_image(image_for_model)
        metadata_batch = None
        if runtime.use_metadata:
            metadata_batch = runtime.encode_metadata(50.0, "unknown", "unknown").unsqueeze(0)
        probs = runtime.predict_proba(image_batch=image_batch, metadata_batch=metadata_batch)
        sorted_idx = probs.argsort()[::-1]
        pred_idx = int(sorted_idx[0])
        pred_label = class_names[pred_idx] if pred_idx < len(class_names) else "unknown"
        top_prob = float(probs[pred_idx])
        gt_upper = gt_label.upper()
        pred_upper = pred_label.upper()
        top2_labels = [class_names[i].upper() for i in sorted_idx[:2] if i < len(class_names)]
        top3_labels = [class_names[i].upper() for i in sorted_idx[:3] if i < len(class_names)]

        prob_dict = runtime.probabilities_dict(probs)
        mel_prob = float(
            prob_dict.get("mel", prob_dict.get("MEL", 0.0))
        )
        correct = pred_upper == gt_upper
        results.append({
            "image_id": image_id,
            "ground_truth_label": gt_label,
            "predicted_label": pred_label,
            "ground_truth_idx": int(CLASS_COLS.index(gt_upper)) if gt_upper in CLASS_COLS else -1,
            "predicted_idx": int(class_to_idx_upper.get(pred_upper, -1)),
            "correct": bool(correct),
            "top2_correct": bool(gt_upper in top2_labels),
            "top3_correct": bool(gt_upper in top3_labels),
            "top_prob": round(top_prob, 4),
            "mel_prob": round(mel_prob, 6),
            **{k: round(prob_dict.get(k, 0.0), 4) for k in class_names},
        })

        if out_images:
            dest = out_images / f"{image_id}_gt_{gt_label}_pred_{pred_label}.jpg"
            try:
                image.save(dest)
            except Exception:
                pass

    # Write report
    args.out_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.out_dir / "validation_report.csv"
    result_df = pd.DataFrame(results)
    result_df.to_csv(report_path, index=False)
    correct = result_df["correct"].sum()
    top2_correct = result_df["top2_correct"].sum()
    top3_correct = result_df["top3_correct"].sum()
    total = len(result_df)
    acc = correct / total if total else 0.0
    acc_top2 = top2_correct / total if total else 0.0
    acc_top3 = top3_correct / total if total else 0.0
    y_true = result_df["ground_truth_label"].astype(str).str.upper().to_numpy()
    y_pred = result_df["predicted_label"].astype(str).str.upper().to_numpy()
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, labels=CLASS_COLS, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=CLASS_COLS, average="weighted", zero_division=0))
    extra_pred_labels = sorted(label for label in set(y_pred.tolist()) if label not in CLASS_COLS)
    cm_labels = CLASS_COLS + extra_pred_labels
    cm = confusion_matrix(y_true, y_pred, labels=cm_labels)

    mel_threshold = float(args.mel_threshold) if args.mel_threshold is not None else float(runtime.mel_threshold)
    y_true_mel = (y_true == "MEL").astype(int)
    y_pred_mel = (result_df["mel_prob"].to_numpy(dtype=float) >= mel_threshold).astype(int)
    mel_tp = int(((y_true_mel == 1) & (y_pred_mel == 1)).sum())
    mel_tn = int(((y_true_mel == 0) & (y_pred_mel == 0)).sum())
    mel_fp = int(((y_true_mel == 0) & (y_pred_mel == 1)).sum())
    mel_fn = int(((y_true_mel == 1) & (y_pred_mel == 0)).sum())
    mel_sensitivity = _safe_div(mel_tp, mel_tp + mel_fn)
    mel_specificity = _safe_div(mel_tn, mel_tn + mel_fp)
    mel_precision = _safe_div(mel_tp, mel_tp + mel_fp)
    mel_npv = _safe_div(mel_tn, mel_tn + mel_fn)

    # Summary by class
    summary_path = args.out_dir / "validation_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"ISIC 2018 validation (app model)\n")
        f.write(f"Bundle: {bundle_path}\n")
        f.write(f"Split: {args.split}\n")
        f.write(f"Preprocess: {args.preprocess}\n")
        f.write(f"Total samples: {total}\n")
        f.write(f"Correct: {correct}\n")
        f.write(f"Top-1 Accuracy: {acc:.2%}\n")
        f.write(f"Top-2 Accuracy: {acc_top2:.2%}\n")
        f.write(f"Top-3 Accuracy: {acc_top3:.2%}\n\n")
        f.write(f"Balanced Accuracy: {bal_acc:.2%}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write("Melanoma triage (one-vs-rest from MEL probability):\n")
        f.write(f"  MEL threshold: {mel_threshold:.4f}\n")
        f.write(f"  Sensitivity: {mel_sensitivity:.2%}\n")
        f.write(f"  Specificity: {mel_specificity:.2%}\n")
        f.write(f"  Precision (PPV): {mel_precision:.2%}\n")
        f.write(f"  NPV: {mel_npv:.2%}\n")
        f.write(f"  Confusion (TP, FP, TN, FN): ({mel_tp}, {mel_fp}, {mel_tn}, {mel_fn})\n\n")
        f.write("Per-class accuracy (ground_truth_label -> correct count / total):\n")
        for label in CLASS_COLS:
            sub = result_df[result_df["ground_truth_label"] == label]
            if len(sub) == 0:
                continue
            c = sub["correct"].sum()
            n = len(sub)
            f.write(f"  {label}: {c}/{n} ({100.0 * c / n:.1f}%)\n")
        f.write("\nIncorrect predictions (image_id, gt -> pred):\n")
        incorrect = result_df[~result_df["correct"]]
        for _, r in incorrect.iterrows():
            f.write(f"  {r['image_id']}: {r['ground_truth_label']} -> {r['predicted_label']}\n")

    metrics_payload = {
        "bundle": str(bundle_path),
        "split": args.split,
        "preprocess": args.preprocess,
        "full_split": bool(args.full_split),
        "total_samples": int(total),
        "top1_accuracy": float(acc),
        "top2_accuracy": float(acc_top2),
        "top3_accuracy": float(acc_top3),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "melanoma_threshold": float(mel_threshold),
        "melanoma_one_vs_rest": {
            "sensitivity": float(mel_sensitivity),
            "specificity": float(mel_specificity),
            "precision_ppv": float(mel_precision),
            "npv": float(mel_npv),
            "tp": mel_tp,
            "fp": mel_fp,
            "tn": mel_tn,
            "fn": mel_fn,
        },
        "labels": cm_labels,
        "confusion_matrix": cm.tolist(),
    }
    metrics_path = args.out_dir / "validation_metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    print(f"Metrics JSON: {metrics_path}")
    print(f"Top-1 Accuracy: {correct}/{total} = {acc:.2%}")
    print(f"Top-2 Accuracy: {top2_correct}/{total} = {acc_top2:.2%}")
    print(f"Top-3 Accuracy: {top3_correct}/{total} = {acc_top3:.2%}")
    print(f"Balanced Accuracy: {bal_acc:.2%}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"MEL Sensitivity @ {mel_threshold:.2f}: {mel_sensitivity:.2%}")
    print(f"MEL Specificity @ {mel_threshold:.2f}: {mel_specificity:.2%}")
    if out_images:
        print(f"Images copied to: {out_images}")


if __name__ == "__main__":
    main()
