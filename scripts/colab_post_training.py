"""Colab post-training pipeline for DermaFusion.

Automates:
1) export bundle from checkpoint
2) validation threshold search on VAL split
3) final VAL/TEST validation with selected threshold
4) clinical readiness report generation
5) artifact packaging
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-training Colab pipeline.")
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Project root path.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best.ckpt",
        help="Checkpoint path from training.",
    )
    parser.add_argument(
        "--bundle-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "export" / "dermafusion_bundle.pt",
        help="Exported bundle path.",
    )
    parser.add_argument(
        "--isic-dir",
        type=Path,
        default=PROJECT_ROOT / "Datasets" / "ISIC2018_Task3",
        help="ISIC 2018 Task 3 directory for validation.",
    )
    parser.add_argument(
        "--preprocess",
        type=str,
        default="sog",
        choices=["runtime", "sog", "sog_hair"],
        help="Preprocess mode used by validate_app_isic2018.py.",
    )
    parser.add_argument(
        "--runtime-color-constancy",
        type=int,
        default=1,
        choices=[0, 1],
        help="Set DERMAFUSION_COLOR_CONSTANCY for runtime transform.",
    )
    parser.add_argument(
        "--initial-export-threshold",
        type=float,
        default=0.30,
        help="MEL threshold embedded into exported bundle (can be overridden during validation).",
    )
    parser.add_argument(
        "--candidate-thresholds",
        type=str,
        default="0.30,0.20,0.15,0.10,0.05,0.02",
        help="Comma-separated MEL threshold candidates evaluated on VAL.",
    )
    parser.add_argument(
        "--target-mel-sensitivity",
        type=float,
        default=0.80,
        help="Target MEL one-vs-rest sensitivity for threshold selection.",
    )
    parser.add_argument(
        "--target-mel-specificity",
        type=float,
        default=0.70,
        help="Target MEL one-vs-rest specificity for threshold selection.",
    )
    parser.add_argument(
        "--target-mel-npv",
        type=float,
        default=0.97,
        help="Target MEL one-vs-rest NPV for threshold selection.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "colab_post_training",
        help="Directory for generated validation/gate artifacts.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Only export bundle and package artifacts (skip threshold search/validation/gates).",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip TEST validation (run VAL only).",
    )
    parser.add_argument(
        "--drive-copy-dir",
        type=Path,
        default=None,
        help="Optional directory (e.g., /content/drive/MyDrive/DermaFusion_runs/run_x) to copy artifacts.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing.",
    )
    parser.add_argument(
        "--drive-root",
        type=Path,
        default=Path("/content/drive/MyDrive"),
        help="Google Drive search root used for auto-discovery after runtime reset.",
    )
    parser.add_argument(
        "--find-checkpoint-in-drive",
        action="store_true",
        help="If --checkpoint is missing, search Drive for a checkpoint automatically.",
    )
    parser.add_argument(
        "--find-isic-in-drive",
        action="store_true",
        help="If --isic-dir is missing, search Drive for ISIC2018_Task3 automatically.",
    )
    parser.add_argument(
        "--no-localize-drive-checkpoint",
        action="store_true",
        help="Use discovered Drive checkpoint in-place instead of copying to outputs/checkpoints/best.ckpt.",
    )
    return parser.parse_args()


def _run(cmd: list[str], cwd: Path, env: dict[str, str], dry_run: bool) -> None:
    cmd_text = " ".join(cmd)
    print(f"[run] {cmd_text}")
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def _load_metrics(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _threshold_token(value: float) -> str:
    text = f"{value:.4f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def _parse_thresholds(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("No valid threshold values provided.")
    return values


def _pick_threshold(
    rows: list[dict[str, Any]],
    min_sens: float,
    min_spec: float,
    min_npv: float,
) -> dict[str, Any]:
    feasible = [
        row
        for row in rows
        if row["mel_sensitivity"] >= min_sens
        and row["mel_specificity"] >= min_spec
        and row["mel_npv"] >= min_npv
    ]
    if feasible:
        # Prefer higher threshold among feasible options to reduce false positives.
        return sorted(feasible, key=lambda x: x["threshold"], reverse=True)[0]

    # Fallback: maximize Youden J on MEL one-vs-rest.
    for row in rows:
        row["youden_j"] = row["mel_sensitivity"] + row["mel_specificity"] - 1.0
    return sorted(rows, key=lambda x: x["youden_j"], reverse=True)[0]


def _discover_latest_file(search_root: Path, patterns: list[str]) -> Path | None:
    if not search_root.exists():
        return None
    seen: dict[Path, float] = {}
    for pattern in patterns:
        for candidate in search_root.rglob(pattern):
            if not candidate.is_file():
                continue
            try:
                key = candidate.resolve()
                seen[key] = candidate.stat().st_mtime
            except Exception:
                continue
    if not seen:
        return None
    best_path = max(seen.items(), key=lambda item: item[1])[0]
    return best_path


def _discover_named_dir(search_root: Path, dir_name: str) -> Path | None:
    if not search_root.exists():
        return None
    matches: list[Path] = []
    for candidate in search_root.rglob(dir_name):
        if candidate.is_dir():
            matches.append(candidate)
    if not matches:
        return None
    matches.sort(key=lambda path: (len(path.parts), str(path)))
    return matches[0]


def _zip_dir(source_dir: Path, output_zip: Path) -> None:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(source_dir.rglob("*")):
            if path.is_dir():
                continue
            if path.resolve() == output_zip.resolve():
                continue
            zf.write(path, arcname=str(path.relative_to(source_dir)))


def main() -> None:
    args = parse_args()
    root = args.project_root.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else (root / args.checkpoint)
    bundle_output = args.bundle_output if args.bundle_output.is_absolute() else (root / args.bundle_output)
    isic_dir = args.isic_dir if args.isic_dir.is_absolute() else (root / args.isic_dir)
    output_dir = args.output_dir if args.output_dir.is_absolute() else (root / args.output_dir)
    drive_root = args.drive_root if args.drive_root.is_absolute() else (root / args.drive_root)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_source = checkpoint
    if not checkpoint.exists() and args.find_checkpoint_in_drive:
        discovered_ckpt = _discover_latest_file(
            drive_root,
            patterns=["best.ckpt", "DermaFusion_best.ckpt", "*.ckpt"],
        )
        if discovered_ckpt is not None:
            checkpoint_source = discovered_ckpt
            checkpoint = discovered_ckpt
            print(f"[discover] checkpoint: {discovered_ckpt}")
            if not args.no_localize_drive_checkpoint and not args.dry_run:
                localized_ckpt = root / "outputs" / "checkpoints" / "best.ckpt"
                localized_ckpt.parent.mkdir(parents=True, exist_ok=True)
                if discovered_ckpt.resolve() != localized_ckpt.resolve():
                    shutil.copy2(discovered_ckpt, localized_ckpt)
                checkpoint = localized_ckpt
                print(f"[localize] checkpoint copied to: {localized_ckpt}")

    if not checkpoint.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint}. "
            "Pass --checkpoint <path> or use --find-checkpoint-in-drive."
        )

    if not args.skip_validation and not isic_dir.exists() and args.find_isic_in_drive:
        discovered_isic = _discover_named_dir(drive_root, "ISIC2018_Task3")
        if discovered_isic is not None:
            isic_dir = discovered_isic
            print(f"[discover] ISIC dir: {isic_dir}")

    if not args.skip_validation and not isic_dir.exists() and not args.dry_run:
        raise FileNotFoundError(
            f"ISIC directory not found: {isic_dir}. "
            "Pass --isic-dir <path> or use --find-isic-in-drive."
        )

    env = os.environ.copy()
    env["DERMAFUSION_COLOR_CONSTANCY"] = str(int(args.runtime_color_constancy))

    # 1) Export bundle from checkpoint.
    _run(
        [
            sys.executable,
            "scripts/export_bundle.py",
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(bundle_output),
            "--mel-threshold",
            str(args.initial_export_threshold),
        ],
        cwd=root,
        env=env,
        dry_run=args.dry_run,
    )

    if args.skip_validation:
        if args.dry_run:
            print("[info] Dry run complete.")
            return
        summary_path = output_dir / "pipeline_summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "checkpoint": str(checkpoint),
                    "checkpoint_source": str(checkpoint_source),
                    "bundle_output": str(bundle_output),
                    "skip_validation": True,
                    "note": "Bundle export completed. Validation/gate steps skipped.",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        zip_path = output_dir.parent / f"{output_dir.name}_artifacts.zip"
        _zip_dir(output_dir, zip_path)
        print(f"[done] summary: {summary_path}")
        print(f"[done] zip: {zip_path}")
        return

    thresholds = _parse_thresholds(args.candidate_thresholds)
    threshold_rows: list[dict[str, Any]] = []

    # 2) Threshold search on VAL split.
    for threshold in thresholds:
        token = _threshold_token(threshold)
        out_dir = output_dir / f"val_thr_{token}"
        _run(
            [
                sys.executable,
                "scripts/validate_app_isic2018.py",
                "--bundle",
                str(bundle_output),
                "--isic-dir",
                str(isic_dir),
                "--split",
                "val",
                "--preprocess",
                args.preprocess,
                "--full-split",
                "--mel-threshold",
                str(threshold),
                "--out-dir",
                str(out_dir),
            ],
            cwd=root,
            env=env,
            dry_run=args.dry_run,
        )
        if args.dry_run:
            continue
        metrics_path = out_dir / "validation_metrics.json"
        metrics = _load_metrics(metrics_path)
        mel = metrics.get("melanoma_one_vs_rest", {})
        threshold_rows.append(
            {
                "threshold": float(metrics.get("melanoma_threshold", threshold)),
                "top1_accuracy": float(metrics.get("top1_accuracy", 0.0)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", 0.0)),
                "macro_f1": float(metrics.get("macro_f1", 0.0)),
                "mel_sensitivity": float(mel.get("sensitivity", 0.0)),
                "mel_specificity": float(mel.get("specificity", 0.0)),
                "mel_npv": float(mel.get("npv", 0.0)),
                "metrics_path": str(metrics_path),
            }
        )

    if args.dry_run:
        print("[info] Dry run complete.")
        return

    if not threshold_rows:
        raise RuntimeError("No threshold search results were generated.")

    selected = _pick_threshold(
        rows=threshold_rows,
        min_sens=args.target_mel_sensitivity,
        min_spec=args.target_mel_specificity,
        min_npv=args.target_mel_npv,
    )
    selected_threshold = float(selected["threshold"])
    selected_token = _threshold_token(selected_threshold)
    print(f"[select] threshold={selected_threshold:.4f}")

    # Save threshold table.
    threshold_table_path = output_dir / "threshold_search.json"
    threshold_table_path.write_text(
        json.dumps(
            {
                "target_mel_sensitivity": args.target_mel_sensitivity,
                "target_mel_specificity": args.target_mel_specificity,
                "target_mel_npv": args.target_mel_npv,
                "rows": threshold_rows,
                "selected": selected,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    # 3) Final VAL/TEST validation with selected threshold.
    val_final_dir = output_dir / "val_final"
    _run(
        [
            sys.executable,
            "scripts/validate_app_isic2018.py",
            "--bundle",
            str(bundle_output),
            "--isic-dir",
            str(isic_dir),
            "--split",
            "val",
            "--preprocess",
            args.preprocess,
            "--full-split",
            "--mel-threshold",
            str(selected_threshold),
            "--out-dir",
            str(val_final_dir),
        ],
        cwd=root,
        env=env,
        dry_run=False,
    )

    test_metrics_path: Path | None = None
    if not args.skip_test:
        test_final_dir = output_dir / "test_final"
        _run(
            [
                sys.executable,
                "scripts/validate_app_isic2018.py",
                "--bundle",
                str(bundle_output),
                "--isic-dir",
                str(isic_dir),
                "--split",
                "test",
                "--preprocess",
                args.preprocess,
                "--full-split",
                "--mel-threshold",
                str(selected_threshold),
                "--out-dir",
                str(test_final_dir),
            ],
            cwd=root,
            env=env,
            dry_run=False,
        )
        test_metrics_path = test_final_dir / "validation_metrics.json"

    # 4) Clinical readiness gate report.
    report_name = f"clinical_readiness_thr_{selected_token}"
    cmd = [
        sys.executable,
        "scripts/clinical_readiness_report.py",
        "--val-metrics",
        str(val_final_dir / "validation_metrics.json"),
        "--out-dir",
        str(output_dir / "reports"),
        "--report-name",
        report_name,
    ]
    if test_metrics_path is not None:
        cmd.extend(["--test-metrics", str(test_metrics_path)])
    _run(cmd, cwd=root, env=env, dry_run=False)

    # 5) Save pipeline summary and zip artifacts.
    summary_path = output_dir / "pipeline_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "checkpoint_source": str(checkpoint_source),
                "bundle_output": str(bundle_output),
                "isic_dir": str(isic_dir),
                "preprocess": args.preprocess,
                "runtime_color_constancy": int(args.runtime_color_constancy),
                "selected_threshold": selected_threshold,
                "threshold_search_path": str(threshold_table_path),
                "val_metrics": str(val_final_dir / "validation_metrics.json"),
                "test_metrics": str(test_metrics_path) if test_metrics_path else None,
                "clinical_report_json": str(output_dir / "reports" / f"{report_name}.json"),
                "clinical_report_md": str(output_dir / "reports" / f"{report_name}.md"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    zip_path = output_dir.parent / f"{output_dir.name}_artifacts.zip"
    _zip_dir(output_dir, zip_path)
    print(f"[done] summary: {summary_path}")
    print(f"[done] zip: {zip_path}")

    if args.drive_copy_dir is not None:
        drive_dir = args.drive_copy_dir if args.drive_copy_dir.is_absolute() else (root / args.drive_copy_dir)
        drive_dir.mkdir(parents=True, exist_ok=True)
        for src in [
            checkpoint,
            bundle_output,
            summary_path,
            zip_path,
            output_dir / "reports" / f"{report_name}.json",
            output_dir / "reports" / f"{report_name}.md",
        ]:
            if src.exists():
                shutil.copy2(src, drive_dir / src.name)
        print(f"[done] copied key artifacts to: {drive_dir}")


if __name__ == "__main__":
    main()
