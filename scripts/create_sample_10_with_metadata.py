"""Create one folder with 10 ISIC 2018 images (different classes), their metadata, and ground truth.

Output folder contains:
- 10 image files (copied from raw data)
- ground_truth_and_metadata.csv: image_id, filename, ground_truth_label, age, sex, localization, dx_type
- README.txt describing the contents

Usage:
  python scripts/create_sample_10_with_metadata.py --isic-dir Datasets/ISIC2018_Task3 --out-dir outputs/sample_10_with_metadata
  python scripts/create_sample_10_with_metadata.py --ham-metadata Datasets/Kaggle_HAM10000/HAM10000_metadata.csv
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CLASS_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create folder with 10 images (different classes), metadata, and ground truth."
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
        default=PROJECT_ROOT / "outputs" / "sample_10_with_metadata",
        help="Output folder for the 10 images and CSV.",
    )
    parser.add_argument(
        "--ham-metadata",
        type=Path,
        default=None,
        help="Optional HAM10000_metadata.csv for age, sex, localization.",
    )
    return parser.parse_args()


def get_ground_truth_label(row: pd.Series) -> str:
    """From ISIC one-hot row return single label (e.g. NV, MEL)."""
    for c in CLASS_COLS:
        if row.get(c, 0) == 1.0:
            return c
    return "unknown"


def resolve_image_path(isic_dir: Path, image_id: str) -> Path | None:
    """Find image in Training_Input, Validation_Input, or Test_Input."""
    for sub in (
        "ISIC2018_Task3_Training_Input",
        "ISIC2018_Task3_Validation_Input",
        "ISIC2018_Task3_Test_Input",
    ):
        folder = isic_dir / sub
        if not folder.exists():
            continue
        for ext in (".jpg", ".jpeg", ".png"):
            p = folder / f"{image_id}{ext}"
            if p.exists():
                return p
    return None


def load_ground_truth(isic_dir: Path) -> pd.DataFrame:
    """Load train/val/test ground truth and add ground_truth_label."""
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
        id_col = "image" if "image" in df.columns else "image_id"
        df["image_id"] = df[id_col].astype(str).str.strip()
        dfs.append(df)
    if not dfs:
        raise FileNotFoundError(f"No ground truth CSVs found under {isic_dir}")
    out = pd.concat(dfs, ignore_index=True)
    out["ground_truth_label"] = out.apply(get_ground_truth_label, axis=1)
    return out


def main() -> None:
    args = parse_args()
    isic_dir = args.isic_dir.resolve()
    if not isic_dir.exists():
        raise FileNotFoundError(f"ISIC directory not found: {isic_dir}")

    gt = load_ground_truth(isic_dir)

    # Resolve image paths
    paths = []
    for _, row in gt.iterrows():
        p = resolve_image_path(isic_dir, row["image_id"])
        paths.append(p)
    gt["image_path"] = paths
    gt = gt[gt["image_path"].notna()].copy()

    # Optional: merge HAM10000 metadata (age, sex, localization)
    ham_path = args.ham_metadata
    if ham_path is None:
        for candidate in (
            PROJECT_ROOT / "Datasets" / "Kaggle_HAM10000" / "HAM10000_metadata.csv",
            PROJECT_ROOT / "Datasets" / "Harvard_Dataverse" / "HAM10000_metadata.csv",
        ):
            if candidate.exists():
                ham_path = candidate
                break
    if ham_path is not None and ham_path.exists():
        ham = pd.read_csv(ham_path)
        if "image_id" in ham.columns:
            ham = ham[["image_id", "age", "sex", "localization", "dx_type"]].drop_duplicates("image_id")
            ham["image_id"] = ham["image_id"].astype(str).str.strip()
            gt = gt.merge(ham, on="image_id", how="left")
        else:
            gt["age"] = pd.NA
            gt["sex"] = ""
            gt["localization"] = ""
            gt["dx_type"] = ""
    else:
        gt["age"] = pd.NA
        gt["sex"] = ""
        gt["localization"] = ""
        gt["dx_type"] = ""

    # Sample 10 images: at least one per class, then fill to 10
    sampled = []
    for label in CLASS_COLS:
        subset = gt[gt["ground_truth_label"] == label]
        if len(subset) > 0:
            sampled.append(subset.sample(n=1, random_state=42))
    sample_df = pd.concat(sampled, ignore_index=True)
    # We have 7 (one per class); add 3 more from first classes to get 10
    need = 10 - len(sample_df)
    if need > 0:
        remaining = gt[~gt["image_id"].isin(sample_df["image_id"])]
        if len(remaining) >= need:
            extra = remaining.sample(n=need, random_state=43)
            sample_df = pd.concat([sample_df, extra], ignore_index=True)
    if len(sample_df) > 10:
        sample_df = sample_df.head(10)

    # Create output folder
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    images_dir = out_dir / "images"
    images_dir.mkdir(exist_ok=True)

    # Copy images and build rows for CSV
    rows = []
    for _, row in sample_df.iterrows():
        src = Path(row["image_path"])
        image_id = row["image_id"]
        label = row["ground_truth_label"]
        filename = f"{image_id}_gt_{label}.jpg"
        dest = images_dir / filename
        if src.exists():
            shutil.copy2(src, dest)
        age = row.get("age", "")
        sex = row.get("sex", "")
        loc = row.get("localization", "")
        dx_type = row.get("dx_type", "")
        if pd.isna(age):
            age = ""
        rows.append({
            "image_id": image_id,
            "filename": filename,
            "ground_truth_label": label,
            "age": age,
            "sex": str(sex).strip() if pd.notna(sex) else "",
            "localization": str(loc).strip() if pd.notna(loc) else "",
            "dx_type": str(dx_type).strip() if pd.notna(dx_type) else "",
        })

    # Write ground truth + metadata CSV
    out_csv = out_dir / "ground_truth_and_metadata.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    # README
    readme = out_dir / "README.txt"
    readme.write_text(
        "Sample of 10 ISIC 2018 images with different classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC).\n\n"
        "Contents:\n"
        "  images/           - 10 image files named {image_id}_gt_{ground_truth_label}.jpg\n"
        "  ground_truth_and_metadata.csv - image_id, filename, ground_truth_label, age, sex, localization, dx_type\n\n"
        "Metadata (age, sex, localization) comes from HAM10000 when available; otherwise empty.\n"
    )

    print(f"Created: {out_dir}")
    print(f"  Images: {images_dir} ({len(rows)} files)")
    print(f"  Ground truth + metadata: {out_csv}")
    print(f"  README: {readme}")


if __name__ == "__main__":
    main()
