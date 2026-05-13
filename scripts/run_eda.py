#!/usr/bin/env python3
"""Run EDA from command line; saves plots to outputs/eda/. Run from repo root: python scripts/run_eda.py"""
from __future__ import annotations

import sys
from pathlib import Path

# Non-interactive backend so plots save to file when run as script
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

sns.set_theme(style="whitegrid")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
META_DIR = RAW_DIR / "metadata"
OUT_DIR = PROJECT_ROOT / "outputs" / "eda"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_CSV = META_DIR / "metadata_merged.csv"
HAM_META_CSV = META_DIR / "HAM10000_metadata.csv"
TRAIN_GT = META_DIR / "ISIC2018_Task3_Training_GroundTruth.csv"
GROUPINGS_CSV = META_DIR / "ISIC2018_Task3_Training_LesionGroupings.csv"

if MERGED_CSV.exists():
    df = pd.read_csv(MERGED_CSV)
    print(f"Loaded metadata: {MERGED_CSV}")
elif HAM_META_CSV.exists():
    df = pd.read_csv(HAM_META_CSV)
    print(f"Loaded metadata: {HAM_META_CSV}")
elif TRAIN_GT.exists() and GROUPINGS_CSV.exists():
    gt = pd.read_csv(TRAIN_GT)
    grp = pd.read_csv(GROUPINGS_CSV)
    df = gt.merge(grp, on="image", how="left").rename(columns={"image": "image_id"})
    print(f"Built metadata from ISIC 2018: {TRAIN_GT.name} + {GROUPINGS_CSV.name}")
else:
    print("No metadata found. Expected metadata_merged.csv, HAM10000_metadata.csv, or ISIC2018 CSVs in data/raw/metadata/", file=sys.stderr)
    sys.exit(1)

raw_images = RAW_DIR / "images"
preprocessed = PROJECT_ROOT / "data" / "preprocessed_hair_removed" / "images"
if (raw_images / "train").exists():
    IMAGE_DIR = raw_images
elif raw_images.exists():
    IMAGE_DIR = RAW_DIR
elif preprocessed.exists():
    IMAGE_DIR = preprocessed
else:
    IMAGE_DIR = raw_images

print(f"Rows: {len(df):,}")
print(df.head().to_string())

# Derive dx if needed
if "dx" not in df.columns and set(["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]).issubset(df.columns):
    class_cols = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    df["dx"] = df[class_cols].idxmax(axis=1).str.lower()

# 1) Class distribution
class_counts = df["dx"].value_counts().sort_values(ascending=False)
class_pct = (class_counts / class_counts.sum()) * 100
summary = pd.DataFrame({"count": class_counts, "percent": class_pct.round(2)})
print("\nClass summary:\n", summary.to_string())
plt.figure(figsize=(10, 5))
sns.barplot(x=summary.index, y=summary["count"], palette="viridis")
plt.title("Class Distribution")
plt.xlabel("Diagnosis")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(OUT_DIR / "01_class_distribution.png", dpi=120)
plt.close()
print(f"Saved {OUT_DIR / '01_class_distribution.png'}")

# 2) Age / sex / localization
if "age" in df.columns:
    plt.figure(figsize=(11, 5))
    sns.boxplot(data=df, x="dx", y="age")
    plt.title("Age Distribution by Diagnosis")
    plt.xlabel("Diagnosis")
    plt.ylabel("Age")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "02_age_by_dx.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR / '02_age_by_dx.png'}")
if "sex" in df.columns:
    sex_dx = pd.crosstab(df["dx"], df["sex"], normalize="index") * 100
    sex_dx.plot(kind="bar", stacked=True, figsize=(11, 5), colormap="Set2")
    plt.title("Sex Distribution by Diagnosis (%)")
    plt.ylabel("Percent")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "03_sex_by_dx.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR / '03_sex_by_dx.png'}")
if "localization" in df.columns:
    loc_dx = pd.crosstab(df["dx"], df["localization"])
    plt.figure(figsize=(14, 6))
    sns.heatmap(loc_dx, cmap="magma")
    plt.title("Diagnosis vs Localization Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "04_localization_heatmap.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR / '04_localization_heatmap.png'}")

# 3) Images per lesion + missing
if "lesion_id" in df.columns:
    lesion_counts = df["lesion_id"].value_counts()
    plt.figure(figsize=(10, 4))
    sns.histplot(lesion_counts, bins=30, kde=False)
    plt.title("Images Per Lesion Distribution")
    plt.xlabel("Images per lesion")
    plt.ylabel("Number of lesions")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "05_images_per_lesion.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR / '05_images_per_lesion.png'}")
missing_pct = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
missing_df = pd.DataFrame({"missing_pct": missing_pct})
plt.figure(figsize=(10, 4))
missing_df["missing_pct"].plot(kind="bar")
plt.title("Missing Value Percentage by Column")
plt.ylabel("% Missing")
plt.tight_layout()
plt.savefig(OUT_DIR / "06_missing_values.png", dpi=120)
plt.close()
print(f"Saved {OUT_DIR / '06_missing_values.png'}")

# 4) Correlation
numeric_cols = [c for c in ["age", "age_norm", "sex_idx", "localization_idx"] if c in df.columns]
if numeric_cols:
    corr = df[numeric_cols].corr(numeric_only=True)
    plt.figure(figsize=(6, 5))
    sns.heatmap(corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
    plt.title("Metadata Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "07_correlation.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR / '07_correlation.png'}")

# 5) Image dimensions + sample grid
train_image_dir = IMAGE_DIR / "train"
if not train_image_dir.exists():
    train_image_dir = IMAGE_DIR
image_files = sorted(train_image_dir.glob("*.jpg"))
if not image_files:
    print(f"No images in {train_image_dir}; skipping image analysis.", file=sys.stderr)
else:
    shapes = []
    for path in image_files[:1000]:
        with Image.open(path) as img:
            w, h = img.size
            arr = np.asarray(img.convert("RGB"))
            variance = float(np.var(arr))
        shapes.append({"file": path.name, "width": w, "height": h, "variance": variance})
    shape_df = pd.DataFrame(shapes)
    print("\nImage shape sample:\n", shape_df[["width", "height", "variance"]].describe().to_string())
    plt.figure(figsize=(10, 4))
    sns.histplot(shape_df["width"], color="steelblue", label="width", bins=30, alpha=0.7)
    sns.histplot(shape_df["height"], color="orange", label="height", bins=30, alpha=0.5)
    plt.legend()
    plt.title("Image Width/Height Distribution (sample)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "08_image_dims.png", dpi=120)
    plt.close()
    plt.figure(figsize=(10, 4))
    sns.histplot(shape_df["variance"], bins=40)
    plt.title("Image Pixel Variance Distribution (sample)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "09_image_variance.png", dpi=120)
    plt.close()
    print(f"Saved {OUT_DIR / '08_image_dims.png'}, {OUT_DIR / '09_image_variance.png'}")

    classes = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
    if "dx" in df.columns:
        fig, axes = plt.subplots(3, 7, figsize=(18, 8))
        for col_idx, cls in enumerate(classes):
            class_rows = df[df["dx"].astype(str).str.lower() == cls].head(3)
            for row_idx in range(3):
                ax = axes[row_idx, col_idx]
                if row_idx < len(class_rows):
                    image_id = str(class_rows.iloc[row_idx]["image_id"])
                    img_path = train_image_dir / f"{image_id}.jpg"
                    if not img_path.exists():
                        img_path = train_image_dir / image_id
                    if img_path.exists():
                        ax.imshow(Image.open(img_path).convert("RGB"))
                        ax.set_title(cls if row_idx == 0 else "")
                    else:
                        ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_DIR / "10_sample_grid.png", dpi=120)
        plt.close()
        print(f"Saved {OUT_DIR / '10_sample_grid.png'}")

print(f"\nEDA done. All figures saved to {OUT_DIR}")
