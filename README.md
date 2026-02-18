# DermaFusion

Multi-modal deep learning for **skin lesion classification** (7 classes: melanoma, nevus, BCC, AKIEC, BKL, DF, vascular). Includes a Python training/evaluation pipeline and an iOS app (DermFusion) for on-device inference.

## Features

- **7-class classification**: MEL, NV, BCC, AKIEC, BKL, DF, VASC (ISIC 2018 / HAM10000)
- **Image-only baseline**: EfficientNet-B4 (timm) with Focal loss and class weighting
- **Lesion-level splits**: No data leakage; train/val/test by `lesion_id`
- **Preprocessing**: Optional hair removal (DullRazor) and color constancy (Shades of Gray)
- **Config-driven**: Hydra configs for model, data, and training
- **Colab-ready**: Notebook for GPU training with hair-removed images

## Setup

**Requirements:** Python 3.10+, PyTorch 2.x, CUDA (optional, for GPU).

```bash
git clone https://github.com/ManikantaSirumalla/DermaFusion.git
cd DermaFusion
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Data

- **Primary:** [ISIC 2018 Task 3](https://challenge.isic-archive.com/landing/2018/) (training/validation/test inputs + ground truth CSVs + lesion groupings).
- Place data under `data/raw/` (see `scripts/setup_data.py` for layout). You need:
  - `data/raw/images/{train,val,test}/` — dermoscopy images
  - `data/raw/metadata/` — ISIC ground truth CSVs and `ISIC2018_Task3_Training_LesionGroupings.csv`

Optional: run preprocessing and hair removal, then train on preprocessed images:

```bash
python scripts/preprocess_data.py --apply-hair-removal
# Output: data/preprocessed_hair_removed/images/{train,val,test}/
```

## Training

**Local (CPU or GPU):**

```bash
python scripts/train.py
```

Override config as needed:

```bash
# Use hair-removed images and CPU (e.g. on Mac without MPS)
python scripts/train.py data.data.use_preprocessed=true \
  data.data.preprocessed_image_dir=data/preprocessed_hair_removed/images \
  training.training.device=cpu

# Short run (2 epochs)
python scripts/train.py training.training.epochs=2
```

**Google Colab (GPU):** Use the notebook `notebooks/Colab_Train_HairRemoved.ipynb`: mount Drive or clone this repo, install deps, then run training with `use_preprocessed=true` and hair-removed image paths.

## Evaluation

After training, the best checkpoint is saved under `outputs/checkpoints/best.ckpt`. Run evaluation:

```bash
python scripts/evaluate.py
```

Reports (JSON/Markdown) are written to `outputs/`.

## Project layout

```
DermaFusion/
├── configs/           # Hydra: model, training, data, experiment
├── scripts/           # train.py, evaluate.py, preprocess_data.py, setup_data.py
├── src/
│   ├── data/          # Dataset, transforms, preprocessing, sampler
│   ├── models/        # Model factory, image-only and fusion architectures
│   ├── training/      # Trainer, losses, optimizer, schedulers, callbacks
│   ├── evaluation/    # Metrics, calibration, confidence intervals
│   └── utils/         # Logging, config, reproducibility
├── notebooks/         # EDA, baseline, Colab training
├── tests/
├── DermFusion/        # iOS app (Swift)
├── requirements.txt
└── README.md
```

## iOS app (DermFusion)

The `DermFusion/` directory contains the Swift/Xcode project for on-device inference. Export the trained model (e.g. Core ML or ONNX) and integrate it into the app for camera-based skin lesion screening.

## License

MIT. See [LICENSE](LICENSE).

## Pushing to GitHub

From the project root:

```bash
git init
git add .
git commit -m "Add DermaFusion training pipeline, configs, and Colab notebook"
git remote add origin https://github.com/ManikantaSirumalla/DermaFusion.git
git branch -M main
git push -u origin main
```

If the remote already has commits (e.g. initial README/LICENSE), either:

- **Overwrite:** `git push -u origin main --force`
- **Merge:** `git pull origin main --allow-unrelated-histories` then resolve any conflicts and `git push -u origin main`

## Reference

- ISIC 2018 Task 3: [Description](https://challenge.isic-archive.com/landing/2018/)
- HAM10000: Tschandl et al., *Scientific Data* (2018)
