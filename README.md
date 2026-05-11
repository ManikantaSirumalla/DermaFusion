# DermaFusion

Multi-modal deep learning for **skin lesion classification** (7 classes: melanoma, nevus, BCC, AKIEC, BKL, DF, vascular). This repository is the **ML side** — training & evaluation pipeline, datasets, notebooks, configs, and experiments.

> 📱 **The iOS app now lives in its own repo:** [ManikantaSirumalla/DermaFusion-App](https://github.com/ManikantaSirumalla/DermaFusion-App) (SwiftUI + Core ML, on-device inference).
> The public legal pages are at [ManikantaSirumalla/DermaFusion-Legal](https://github.com/ManikantaSirumalla/DermaFusion-Legal).

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
- **Raw ISIC 2018 location:** You can place the downloaded archive under `Datasets/ISIC2018_Task3/`. Expected layout:
  - `Datasets/ISIC2018_Task3/ISIC2018_Task3_Training_Input/` (and Validation/Test) — images
  - `Datasets/ISIC2018_Task3/ISIC2018_Task3_*_GroundTruth/` or `*.csv` at root — ground truth
  - `Datasets/ISIC2018_Task3/ISIC2018_Task3_Training_LesionGroupings.csv`
- **Canonical layout after setup:** Run `scripts/setup_data.py --raw-data-dir Datasets/ISIC2018_Task3 --output-dir data/raw` to get:
  - `data/raw/images/{train,val,test}/` — dermoscopy images
  - `data/raw/metadata/` — ISIC ground truth CSVs and lesion groupings (and optional merged metadata)

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

### Bundle Validation (Colab-Friendly)

If you trained in Colab and only have `dermafusion_bundle.pt`, run full-split validation directly from the bundle:

```bash
# Full validation split with dermoscopy preprocessing (Shades of Gray)
python scripts/validate_app_isic2018.py \
  --bundle dermafusion_bundle.pt \
  --split val \
  --preprocess sog \
  --full-split \
  --out-dir outputs/validation_app_val_sog_fullclin
```

This writes:
- `validation_report.csv` (per-image predictions)
- `validation_summary.txt` (Top-k + per-class metrics + MEL triage sensitivity/specificity)
- `validation_metrics.json` (machine-readable metrics + confusion matrix)

### Clinical Readiness Gates

Generate a pass/fail readiness report from validation metrics:

```bash
python scripts/clinical_readiness_report.py \
  --val-metrics outputs/validation_app_val_sog_fullclin/validation_metrics.json \
  --test-metrics outputs/validation_app_test_sog_fullclin/validation_metrics.json \
  --out-dir outputs/reports \
  --report-name clinical_readiness
```

Outputs:
- `outputs/reports/clinical_readiness.md`
- `outputs/reports/clinical_readiness.json`

Default gate profile (`screening_v1`) checks:
- sample size
- top-1 accuracy, balanced accuracy, macro-F1
- melanoma triage sensitivity/specificity/NPV
- class recall for MEL, BCC, AKIEC

## Deployment Sync (Colab <-> Repo)

To keep Colab artifacts and repo inference in sync, use a shared bundle:

1. Place/export bundle at `outputs/export/dermafusion_bundle.pt`
2. Use shared runtime via:
   - CLI: `python scripts/predict.py --image path/to/image.jpg`
   - Gradio: `python app/gradio_demo.py`

Bundle-first loading is automatic. Fallback is `outputs/checkpoints/best.ckpt`.
Auto-discovered bundle paths include:
- `demo/dermafusion_new_V2/final_t020/dermafusion_bundle.pt`
- `demo/dermafusion_new_V2/export_v2/dermafusion_bundle_t020.pt`
- `demo/dermafusion_new_V2/export_v2/dermafusion_bundle.pt`
- latest `demo/dermafusion_new_V2/run_*/dermafusion_bundle.pt`
- `outputs/export/dermafusion_bundle.pt`
- `dermafusion_bundle.pt` (project root)
- `dermaduaion_bundle.pt` (backward-compatible typo fallback)

### Export Bundle From Checkpoint

```bash
python scripts/export_bundle.py \
  --checkpoint outputs/checkpoints/best.ckpt \
  --output outputs/export/dermafusion_bundle.pt \
  --mel-threshold 0.30
```

### Final Report + Model Card

```bash
python scripts/final_report.py
```

Generates:
- `outputs/reports/final_report.md`
- `outputs/reports/model_card.md`
- `outputs/reports/final_report.json`

### Optional Bundle/Checkpoint Parity Check

```bash
python scripts/check_bundle_parity.py --image-dir data/preprocessed/images/val --samples 32
```

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
