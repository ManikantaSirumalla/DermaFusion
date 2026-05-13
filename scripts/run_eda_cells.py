#!/usr/bin/env python3
"""Run key EDA notebook cells to verify they execute (path resolution, load metadata)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

def run_cell1():
    """Path resolution (first cell)."""
    cwd = Path.cwd().resolve()
    proj = None
    env_root = os.environ.get("DERMAFUSION_ROOT") or os.environ.get("PROJECT_ROOT")
    if env_root and Path(env_root).exists():
        proj = Path(env_root).resolve()
    if not proj:
        for p in [cwd, *cwd.parents]:
            if (p / "src").exists() and (p / "notebooks").exists():
                proj = p
                break
        if not proj:
            for p in [cwd, *cwd.parents]:
                if (p / "src").exists():
                    proj = p
                    break
    proj = proj or cwd
    proj = Path(proj).resolve()
    data_raw = proj / "data" / "raw"
    meta_dir = data_raw / "metadata"
    has_meta = (meta_dir / "metadata_merged.csv").exists() or (
        meta_dir / "ISIC2018_Task3_Training_GroundTruth.csv"
    ).exists() if meta_dir.exists() else False
    print("[Cell 1] Project root:", proj)
    print("[Cell 1] data/raw exists:", data_raw.exists())
    print("[Cell 1] metadata:", has_meta)
    return proj, data_raw, meta_dir


def run_cell2(project_root: Path, raw_dir: Path, meta_dir: Path):
    """Load metadata (second cell logic)."""
    import pandas as pd

    CLASS_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]

    def _ensure_dx(df):
        if "dx" in df.columns:
            return df
        df = df.copy()
        df["dx"] = df[CLASS_COLS].idxmax(axis=1).str.lower()
        return df

    def _ensure_image_id(df):
        if "image_id" not in df.columns and "image" in df.columns:
            return df.rename(columns={"image": "image_id"})
        return df

    merged = meta_dir / "metadata_merged.csv"
    val_gt = meta_dir / "ISIC2018_Task3_Validation_GroundTruth.csv"
    test_gt = meta_dir / "ISIC2018_Task3_Test_GroundTruth.csv"
    if not merged.exists() and not (meta_dir / "ISIC2018_Task3_Training_GroundTruth.csv").exists():
        print("[Cell 2] No metadata CSVs found")
        return None

    parts = []
    if merged.exists():
        train_df = pd.read_csv(merged)
    else:
        train_df = pd.read_csv(meta_dir / "ISIC2018_Task3_Training_GroundTruth.csv")
    train_df = _ensure_dx(_ensure_image_id(train_df))
    train_df["split"] = "train"
    parts.append(train_df)
    if val_gt.exists():
        v = pd.read_csv(val_gt)
        v = _ensure_dx(_ensure_image_id(v))
        v["split"] = "val"
        parts.append(v)
    if test_gt.exists():
        t = pd.read_csv(test_gt)
        t = _ensure_dx(_ensure_image_id(t))
        t["split"] = "test"
        parts.append(t)
    df = pd.concat(parts, ignore_index=True)
    print("[Cell 2] Loaded rows:", len(df))
    print("[Cell 2] Splits:", df["split"].value_counts().to_dict())
    return df


def main():
    print("Running EDA notebook cells...")
    proj, raw_dir, meta_dir = run_cell1()
    df = run_cell2(proj, raw_dir, meta_dir)
    if df is not None:
        print("[OK] Class distribution (top 3):", df["dx"].value_counts().head(3).to_dict())
    print("Done.")


if __name__ == "__main__":
    main()
