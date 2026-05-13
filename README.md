# DermaFusion

Multi-source deep learning for **8-class skin-lesion classification** (MEL, NV, BCC, AKIEC, BKL, DF, VASC, OTHER), trained on four public dermatology datasets and deployed as both a Gradio research demo and a native iOS app.

> 📱 **iOS app:** [ManikantaSirumalla/DermaFusion-App](https://github.com/ManikantaSirumalla/DermaFusion-App) (SwiftUI + Core ML, on-device inference).
> ⚖️ **Public legal pages:** [ManikantaSirumalla/DermaFusion-Legal](https://github.com/ManikantaSirumalla/DermaFusion-Legal).

This repository is the **ML side**: training pipeline, Colab notebook, Gradio inference runtime, CoreML export, and the deployment artifacts shipped to the iOS app.

## Headline results

Final E0 + E3 soft-vote ensemble (EfficientNet-B4 × 2, 380 × 380), TEST set n = 5,279, 1,000-iter bootstrap 95 % CI:

| Metric | Value (95 % CI) |
|---|---|
| 8-class Balanced Accuracy | 0.540 [0.509, 0.568] |
| 8-class Macro F1 | 0.469 [0.441, 0.495] |
| Top-1 / Top-2 / Top-3 accuracy | 0.778 / 0.907 / 0.954 |
| Binary-malignant AUROC | 0.830 |
| Sensitivity @ 95 % specificity | 0.462 |

Full table, per-class breakdown, and ISIC 2019 leaderboard comparison: [presentation/slides.md](presentation/slides.md) and `notebooks/DermaFusion_Personal _Final.ipynb` (Phase 15 + 20).

## What's in this repo

```
DermaFusion/
├── notebooks/
│   └── DermaFusion_Personal _Final.ipynb   # The end-to-end training notebook (Colab)
├── src/
│   ├── data/         # Dataset, transforms (incl. Shades-of-Gray + DullRazor), sampler
│   ├── models/       # timm-backed model factory + metadata fusion head
│   ├── training/     # Trainer, CB-Focal loss, EMA/SAM, schedulers
│   ├── evaluation/   # Metrics, calibration, GradCAM
│   ├── deployment/   # ✦ Inference runtime + DermaFusionEnsemble class
│   └── utils/
├── scripts/          # train.py, evaluate.py, predict.py, validate_app_isic2018.py,
│                     # clinical_readiness_report.py, export_bundle.py,
│                     # export_ensemble_for_deployment.py, ...
├── configs/          # Hydra: data, training, model
├── demo/
│   └── app.py        # Gradio research demo (bundle-first inference runtime)
├── app/
│   └── deployment_Bundle/
│       ├── export_coreml.py        # Mac-side CoreML export script
│       └── ensemble_config.json    # Ensemble weights + per-class thresholds
├── presentation/
│   └── slides.md     # 15-slide deck (Marp/reveal-md/slidev compatible)
├── requirements.txt
└── README.md
```

## Architecture summary

| Stage | Description |
|---|---|
| **Data** | ISIC 2018 / 2019 / 2020 + PAD-UFES-20 → 61,694 dermoscopic + smartphone images, 8 unified classes |
| **Splits** | Patient-grouped (`GroupShuffleSplit` on `patient_id → lesion_id`) + iterative leakage enforcement → TRAIN 52,483 / VAL 3,932 / TEST 5,279, **zero patient/lesion overlap** |
| **Preprocessing** | Shades-of-Gray (power = 6) + DullRazor + 380 × 380 (cached as a 2.1 GB tarball on Drive) |
| **Backbone** | EfficientNet-B4 (~17.6 M params) — two members trained from different starting points |
| **E0** | `efficientnet_b4`, CB-Focal (γ = 1.5) + MixUp + TrivialAugmentWide + EMA 0.999, √-inv sampler |
| **E3** | `tf_efficientnet_b4.ns_jft_in1k`, class-balanced focal (γ = 2.0), full-inverse sampler |
| **Ensemble** | Soft-vote, Dirichlet-optimised weights `[0.705, 0.295]`, per-class thresholds tuned on VAL macro-F1 |
| **Deployment** | Gradio (single 142 MB `dermafusion_ensemble.pt`) + iOS (two FP16 `.mlpackage` files, 34 MB each) |

## Setup

```bash
git clone https://github.com/ManikantaSirumalla/DermaFusion.git
cd DermaFusion
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Python 3.10+**, PyTorch 2.1+, optionally CUDA for training. On Apple Silicon make sure `coremltools`, `rpds-py`, and `pydantic-core` are installed as native `arm64` wheels (force-reinstall if you previously had x86_64 wheels under Rosetta).

## Deployment artifacts (large binaries are not in the repo)

GitHub rejects files > 100 MB. The trained checkpoints, ensemble bundle, and CoreML packages live outside the repo. To run inference locally you need to obtain them and place them at one of the auto-discovered paths.

| Artifact | Size | Where the runtime looks for it |
|---|---|---|
| `dermafusion_ensemble.pt` (Gradio) | ~142 MB | `app/deployment_Bundle/dermafusion_ensemble.pt` (preferred), or set `DERMAFUSION_BUNDLE=...` |
| `E0.pt` / `E3.pt` (per-member ckpts) | ~71 MB each | `app/deployment_Bundle/` (only needed if regenerating CoreML packages) |
| `E0.mlpackage` / `E3.mlpackage` | ~34 MB each | `app/deployment_Bundle/` (consumed by the iOS app via the [companion repo](https://github.com/ManikantaSirumalla/DermaFusion-App)) |

Two ways to get them:

1. **Train them yourself** in Colab via `notebooks/DermaFusion_Personal _Final.ipynb` (Phases 1 – 14). The notebook auto-archives `dermafusion_ensemble.pt` to Google Drive.
2. **Download from a GitHub Release** of this repo when one is published.

Once the `.pt` bundle is at `app/deployment_Bundle/dermafusion_ensemble.pt`, the Gradio runtime auto-detects it. To regenerate the CoreML packages from `E0.pt + E3.pt + dermafusion_ensemble.pt`:

```bash
cd app/deployment_Bundle
python export_coreml.py
# → produces E0.mlpackage, E3.mlpackage, ensemble_config.json
```

## Run the Gradio demo

```bash
python -m demo.app
```

The runtime auto-discovers the bundle. On first request it cold-loads `DermaFusionEnsemble` (the pickled `nn.Module` shipped in the `.pt`); subsequent predictions reuse the warm model. UI shows top-K probabilities, GradCAM, and MEL triage decision.

Override the bundle location:

```bash
DERMAFUSION_BUNDLE=/path/to/your/bundle.pt python -m demo.app
```

## Train from scratch (Colab recommended)

The full pipeline lives in `notebooks/DermaFusion_Personal _Final.ipynb`. The notebook header documents a strict run order on a fresh kernel (Phase 0.1 → 9.5 → 9.6 → 14 → 15 → 16 → 20). One A100 / L4 takes ~96 min for E0 and ~6 h for E3.

Local CLI training (single-model B4) is still supported:

```bash
# Smoke test
python scripts/train.py training.training.epochs=2

# Full run with hair-removed images on CPU
python scripts/train.py data.data.use_preprocessed=true \
  data.data.preprocessed_image_dir=data/preprocessed_hair_removed/images \
  training.training.device=cpu
```

After training, evaluate and (optionally) export a deployment bundle:

```bash
python scripts/evaluate.py
python scripts/export_bundle.py \
  --checkpoint outputs/checkpoints/best.ckpt \
  --output outputs/export/dermafusion_bundle.pt \
  --mel-threshold 0.30
```

## Bundle validation + clinical readiness gates

Run full-split validation against a `.pt` bundle (Colab-friendly — no Hydra config required):

```bash
python scripts/validate_app_isic2018.py \
  --bundle app/deployment_Bundle/dermafusion_ensemble.pt \
  --split val --preprocess sog --full-split \
  --out-dir outputs/validation
```

Then turn the JSON output into a pass/fail readiness report:

```bash
python scripts/clinical_readiness_report.py \
  --val-metrics  outputs/validation/validation_metrics.json \
  --test-metrics outputs/validation_test/validation_metrics.json \
  --out-dir outputs/reports --report-name clinical_readiness
```

The default `screening_v1` profile checks sample size, top-1 / balanced accuracy / macro-F1, MEL triage sensitivity / specificity / NPV, and class recall for MEL / BCC / AKIEC.

## Presentation

A 15-slide overview deck lives at [presentation/slides.md](presentation/slides.md). Render with:

```bash
marp presentation/slides.md -o slides.pdf          # PDF
npx reveal-md presentation/slides.md               # HTML (reveal.js)
npx slidev presentation/slides.md                  # Slidev
```

## References

- ISIC 2018 Task 3 — https://challenge.isic-archive.com/landing/2018/
- ISIC 2019 — https://challenge.isic-archive.com/landing/2019/
- HAM10000 — Tschandl et al., *Scientific Data* (2018)
- PAD-UFES-20 — Pacheco et al., *Data in Brief* (2020)
- Shades-of-Gray colour constancy — Finlayson & Trezzi (2004)
- DullRazor hair removal — Lee et al. (1997)
- Class-balanced focal loss — Cui et al., *CVPR* (2019)

## License

MIT — see [LICENSE](LICENSE).
