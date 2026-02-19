# DermaFusion – Colab setup (step-by-step)

Run training on Google Colab with GPU. Follow these steps in order.

---

## Step 1: Open Colab and turn on GPU

1. Go to [Google Colab](https://colab.research.google.com).
2. **File → New notebook** (or open `notebooks/Colab_Train_HairRemoved.ipynb` from the repo).
3. **Runtime → Change runtime type → Hardware accelerator: GPU** (e.g. T4) → **Save**.

---

## Step 2: Clone the repo

In the first cell, run:

```python
!git clone https://github.com/ManikantaSirumalla/DermaFusion.git /content/DermaFusion
%cd /content/DermaFusion
```

Run the cell (Shift+Enter). You should see the repo clone and the working directory change to `/content/DermaFusion`.

---

## Step 3: Install dependencies

In a new cell:

```python
!pip install -q -r requirements.txt
print("Done.")
```

Run it. Wait until it finishes (usually 1–2 minutes).

---

## Step 4: Put your data in the right place

The code expects this layout **inside** `/content/DermaFusion/`:

```
DermaFusion/
  data/
    raw/
      metadata/     ← ISIC 2018 CSVs here
    preprocessed_hair_removed/
      images/
        train/      ← hair-removed training images (.jpg)
        val/        ← hair-removed validation images (.jpg)
```

**Option A – Data on Google Drive**

1. Upload your `data/` folder (with `raw/metadata/` and `preprocessed_hair_removed/images/train`, `val`) to Drive, e.g. `MyDrive/DermaFusion_data/`.
2. In Colab, run:

```python
from google.colab import drive
drive.mount("/content/drive")

# Copy data into the cloned project
!cp -r "/content/drive/MyDrive/DermaFusion_data/"* /content/DermaFusion/data/
```

Adjust `DermaFusion_data` if your Drive folder name or path is different.

**Option B – Upload `derma_data.zip` from your computer**

1. In Colab: **Files** (left sidebar) → **Upload to session storage** → upload `derma_data.zip`.
2. Run this in a cell (adjust the zip path if you used a different name/location):

```python
from pathlib import Path
import zipfile

repo = Path("/content/DermaFusion")
zip_path = Path("/content/derma_data.zip")
if not zip_path.exists():
    zip_path = repo / "derma_data.zip"
if not zip_path.exists():
    raise FileNotFoundError("Upload derma_data.zip to /content/ or set zip_path.")

with zipfile.ZipFile(zip_path, "r") as z:
    names = z.namelist()
# If zip has top-level "data/", extract into repo root
if names and names[0].startswith("data/"):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(repo)
else:
    # Zip has e.g. raw/ and preprocessed_hair_removed/ at top level
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall("/tmp/derma_extract")
    (repo / "data").mkdir(parents=True, exist_ok=True)
    import shutil
    for name in ["raw", "preprocessed_hair_removed"]:
        src = Path("/tmp/derma_extract") / name
        if src.exists():
            shutil.copytree(src, repo / "data" / name, dirs_exist_ok=True)
    shutil.rmtree("/tmp/derma_extract", ignore_errors=True)

# Verify
meta = repo / "data" / "raw" / "metadata"
train_img = repo / "data" / "preprocessed_hair_removed" / "images" / "train"
val_img = repo / "data" / "preprocessed_hair_removed" / "images" / "val"
print("metadata dir:", meta.exists(), list(meta.glob("*.csv"))[:5] if meta.exists() else [])
print("train images:", train_img.exists(), len(list(train_img.glob("*.*"))) if train_img.exists() else 0)
print("val images:", val_img.exists(), len(list(val_img.glob("*.*"))) if val_img.exists() else 0)
```

3. If your zip uses a different layout (e.g. `preprocessed/images` instead of `preprocessed_hair_removed/images`), either rename the folder inside `data/` to `preprocessed_hair_removed/images` or pass `data.data.preprocessed_image_dir=data/preprocessed/images` when training.

**Option C – Upload a generic data zip**

If you uploaded a zip named e.g. `data.zip` that already contains a top-level `data/` folder:

```python
!unzip -o /content/data.zip -d /content/DermaFusion/
```

**Required metadata files (in `data/raw/metadata/`):**

- `ISIC2018_Task3_Training_GroundTruth.csv`
- `ISIC2018_Task3_Validation_GroundTruth.csv` (or similar Validation CSV)
- `ISIC2018_Task3_Test_GroundTruth.csv` (or similar Test CSV)
- `ISIC2018_Task3_Training_LesionGroupings.csv`

**Required images:**  
`data/preprocessed_hair_removed/images/train/` and `.../val/` with `.jpg` files whose names match the `image_id` in the ground truth CSVs (e.g. `ISIC_xxxxx.jpg`).

---

## Step 4b: Run EDA (optional, before training)

To explore class distribution, age/sex/localization, and sample images:

1. In Colab: **File → Open notebook** → open `notebooks/01_eda.ipynb` from the repo (or upload it).
2. Set the notebook’s working directory to the repo (e.g. first cell or **Runtime → Change runtime type** already with repo at `/content/DermaFusion`).
3. Run all cells.

The EDA notebook will use:

- **Metadata:** `data/raw/metadata/metadata_merged.csv`, or `HAM10000_metadata.csv`, or it will build a table from `ISIC2018_Task3_Training_GroundTruth.csv` + `ISIC2018_Task3_Training_LesionGroupings.csv` if those are the only files present.
- **Images:** `data/raw/images/train/` if it exists, otherwise `data/preprocessed_hair_removed/images/` (so it works with hair-removed data in Colab).

If you opened the notebook from the repo, ensure the kernel’s cwd is `/content/DermaFusion` (e.g. run `%cd /content/DermaFusion` in a cell before the rest).

---

## Step 5: Check GPU and data

In a new cell:

```python
import torch
from pathlib import Path

print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

root = Path("/content/DermaFusion")
meta = root / "data" / "raw" / "metadata"
train_img = root / "data" / "preprocessed_hair_removed" / "images" / "train"
val_img = root / "data" / "preprocessed_hair_removed" / "images" / "val"
print("Metadata exists:", meta.exists())
print("Train images exist:", train_img.exists())
print("Val images exist:", val_img.exists())
if train_img.exists():
    n = len(list(train_img.glob("*.jpg")))
    print(f"Train image count: {n}")
```

- **CUDA: True** and **GPU: T4** (or similar) → good.  
- **Metadata/Train/Val exist: True** and train count > 0 → you can train.

---

## Step 6: Run training

In a new cell:

```python
!cd /content/DermaFusion && python scripts/train.py \
  data.data.use_preprocessed=true \
  data.data.preprocessed_image_dir=data/preprocessed_hair_removed/images \
  data.data.num_workers=2 \
  data.data.validate_images_on_init=false \
  training.training.epochs=100
```

- Uses **hair-removed** images and **GPU** (no `device=cpu`).
- **100 epochs** with early stopping; reduce `epochs` (e.g. `training.training.epochs=5`) for a quick test.

Let it run. You should see logs like “Using device: cuda”, class weights, then epoch 1, 2, …

---

## Step 7: Save the checkpoint (optional)

When training finishes (or you stop it), copy the best checkpoint to Drive so it isn’t lost when the runtime disconnects:

```python
from google.colab import drive
from pathlib import Path
import shutil

drive.mount("/content/drive")
ckpt = Path("/content/DermaFusion/outputs/checkpoints/best.ckpt")
if ckpt.exists():
    dest = Path("/content/drive/MyDrive/DermaFusion_best.ckpt")
    shutil.copy(ckpt, dest)
    print("Saved to", dest)
else:
    print("No best.ckpt found")
```

---

## Quick reference – all cells in order

| Order | What to run |
|-------|-------------|
| 1 | `!git clone https://github.com/ManikantaSirumalla/DermaFusion.git /content/DermaFusion` then `%cd /content/DermaFusion` |
| 2 | `!pip install -q -r requirements.txt` |
| 3 | Mount Drive and copy `data/` (or upload and unzip data) into `/content/DermaFusion/data/` |
| 4 | Check GPU + data paths (Step 5 code) |
| 5 | Run `scripts/train.py` with the flags in Step 6 |
| 6 | (Optional) Copy `best.ckpt` to Drive (Step 7) |

If something fails, check: (1) Runtime is GPU, (2) `data/raw/metadata/` has the four CSVs, (3) `data/preprocessed_hair_removed/images/train/` and `val/` exist and contain the expected images.
