<!--
DermaFusion — 15-slide deck
Renders cleanly in Marp / reveal.js / slidev / Quarto. Each `---` is a slide.
Numbers are pulled from the v_final notebook run
(`notebooks/DermaFusion_Personal _Final.ipynb`, cells 130 + 134) and the
deployment manifest (`app/deployment_Bundle/ensemble_config.json`).
-->

---

# DermaFusion

### Multi-source skin-lesion classification, deployed on iPhone

**Pipeline:** 4 public dermatology datasets → unified preprocessing → EfficientNet-B4 ensemble → CoreML on-device inference.

**Headline result (TEST, n = 5,279):**
Balanced accuracy **0.540 [0.509, 0.568]** · Macro F1 **0.469 [0.441, 0.495]** · Top-3 accuracy **95.4 %** · Binary-malignant AUROC **0.830**.

**Deployed:** Gradio research demo + native iOS app (E0 + E3 mlpackages, ~70 MB, real-time inference).

---

# Problem & motivation

**Skin cancer screening is a triage problem, not a binary one.**

Most skin-lesion CNNs are trained on a single, dermatoscope-captured public dataset (HAM10000 / ISIC 2018) and tested on the same domain. Two consequences:

1. **Class collapse:** majority-NV bias hides malignant minorities (DF, VASC, SCC) that account for the most clinically expensive misses.
2. **Domain gap:** an iPhone in a clinic doesn't look like a dermoscope. Models trained on lab images degrade sharply on smartphone images of the same lesion.

**DermaFusion goal:** an 8-class classifier that survives both problems and runs on-device, so it can be embedded in a screening flow that a non-specialist actually uses.

---

# Pipeline overview

```
ISIC 2018 ─┐
ISIC 2019 ─┼──▶ Unify ──▶ Splits  ──▶ Preprocess ──▶ Train (E0, E3) ──▶ Ensemble ──▶ Deploy
ISIC 2020 ─┤   (61,694    (no patient   (SoG +         (EfficientNet-B4)   (soft-vote   (Gradio +
PAD-UFES-20┘    images)    leakage)      DullRazor)                          + per-class  CoreML iOS)
                                         + 380×380                           thresholds)
```

| Phase | What happens |
|---|---|
| 1 — Acquire | Download 4 public dermatology datasets (~70 GB raw) |
| 2 — Unify | Normalize 4 schemas → single 8-class label space |
| 3 — Split | Patient-grouped train / val / test, iterative leakage enforcement |
| 4 — Preprocess | Shades-of-Gray color constancy + DullRazor hair removal + 380×380 |
| 5 — EDA | 14 paper-grade figures (class, domain, demographics, t-SNE, correlation) |
| 6–9 — Train | CB-Focal + EMA + MixUp + cosine schedule (E0); class-balanced focal + balanced sampler (E3) |
| 14 — Ensemble | Soft-vote with Dirichlet-optimized weights + per-class thresholds |
| 15–18 — Deploy | Gradio (.pt bundle) + iOS (FP16 .mlpackage × 2) |

---

# Datasets

| Source | Images (after dedup) | Type | Reason for inclusion |
|---|---:|---|---|
| ISIC 2018 Task 3 | 11,720 | Dermatoscopic | Official 7-class benchmark, has train/val/test split |
| ISIC 2019 | 15,316 | Dermatoscopic | Largest public derm set, fills minority classes |
| ISIC 2020 | 33,126 | Dermatoscopic | Adds melanoma diversity (binary MEL/NV labels) |
| PAD-UFES-20 | 1,532 | **Smartphone** | The only real-world-domain dataset in the mix |
| **Unified clean** | **61,694** | mixed | After cross-source dedup + image validation |

**Class harmonization** maps each native taxonomy onto an 8-class space:
`MEL · NV · BCC · AKIEC · BKL · DF · VASC · OTHER`
(e.g. ISIC 2019 `AK → AKIEC`; PAD-UFES `NEV → NV`, `SEK → BKL`; out-of-vocab → `OTHER`.)

10,015 ISIC 2018 image IDs that resurface in ISIC 2019 / 2020 are removed to prevent test contamination.

---

# Class distribution & domain skew

| Class | TRAIN | VAL | TEST |
|---|---:|---:|---:|
| MEL   | 4,495 |   338 |   492 |
| NV    | 39,760 | 3,075 | 3,858 |
| BCC   | 3,234 |   246 |   365 |
| AKIEC | 1,313 |    69 |   109 |
| BKL   | 2,604 |   151 |   310 |
| DF    |   222 |     5 |    57 |
| VASC  |   246 |     7 |    39 |
| OTHER |   609 |    41 |    49 |

**Imbalance ratio (max/min count):** 179× in TRAIN, 615× in VAL, 99× in TEST.

**Domain split (TEST):** 5,180 dermatoscopic / 99 smartphone — the smartphone tail is *thin* and statistically noisy, but it's exactly where deployment matters.

**Smartphone class coverage** is uneven by design (PAD-UFES-20 only):
AKIEC 25.8 % · OTHER 30.8 % · BCC 11.3 % · BKL 7.6 % · MEL 0.7 % · NV 0.5 % · DF / VASC 0 %.

---

# Preprocessing

Each raw image runs a **deterministic 3-step pipeline** before any training or inference touches it:

```
Raw → Resize(380×380, INTER_AREA) → Shades-of-Gray CC(power=6) → DullRazor → JPEG q=95
```

- **Shades-of-Gray (Finlayson & Trezzi 2004):** estimates illuminant per channel and rescales so cross-source color casts collapse. Cross-source standard deviation of mean R/G/B drops by ~30 %, verified in EDA §5.9.
- **DullRazor (Lee et al. 1997):** morphological blackhat → threshold → inpaint to remove hair artifacts. Hair coverage in the wild ranges from 2 % (face) to 25 % (back) of pixels.

Cached as a 2.1 GB tarball on Drive so subsequent runs skip ~15 min of preprocessing.

In iOS, the SAME color-constancy + hair-removal step is reproduced in Swift before the CoreML call — preprocessing parity matters more than fp16 weight precision.

---

# Splits & leakage prevention

**Naive random splits leak.** Two images of the same lesion (different angles, different visits) on either side of the train/test boundary inflate test accuracy.

**Defense:**
1. ISIC 2018 keeps its **official** train/val/test split (preserves benchmark comparability).
2. All other sources split 85 / 7.5 / 7.5 with `GroupShuffleSplit` keyed by `patient_id → lesion_id → image_id`.
3. **Iterative leakage enforcement:** after the initial split, any `lesion_id` or `patient_id` appearing in multiple splits is force-moved back to train. Loop until convergence.

Run output:
```
Iter 1: lesion_conflicts=26 (moved 32) | patient_conflicts=9 (moved 18)
Iter 2: lesion_conflicts=1  (moved  1) | patient_conflicts=0 (moved  0)
Iter 3: lesion_conflicts=0  (moved  0) | patient_conflicts=0 (moved  0)
Converged ✅
```

**Final splits — TRAIN 52,483 · VAL 3,932 · TEST 5,279, zero patient/lesion overlap across any split pair.**

---

# Architecture: EfficientNet-B4 × 2 ensemble

Two complementary B4-class members trained from different starting points and loss surfaces:

| Member | Backbone | Loss | Sampler | Best val score | Role |
|---|---|---|---|---|---|
| **E0** | `efficientnet_b4` | CB-Focal (γ=1.5) + label smoothing 0.05 + MixUp + TrivialAug | √-inv frequency | 0.562 | Calibrated baseline |
| **E3** | `tf_efficientnet_b4.ns_jft_in1k` (Noisy Student) | Class-balanced focal (γ=2.0) | full inverse freq (aggressive) | 0.474 | Rare-class specialist |

Both at **380 × 380**, ~17.6 M params each, trained for 30 epochs with EMA decay 0.9995 / 0.999, AdamW + cosine schedule (warmup 1 epoch).

**Why two B4s and not B4 + ConvNeXt?** Tested in the notebook (E1 = ConvNeXt-Tiny @ 288, E2 = EfficientNet-B5 @ 456) — error correlation was high enough that the marginal benefit of adding more architectures was below the deployment cost on iPhone. Two B4s give the largest accuracy / on-device-size ratio.

---

# Training methodology — what made the difference

The first attempt plateaued at val balanced accuracy 0.55 (predicting NV for everything). What unstuck it:

1. **CB-Focal alpha with the boost applied AFTER clip** (Phase 8.2). Earlier the MEL/BCC/AKIEC clinical-safety boost was nullified by an upstream lower clip — fix gave MEL alpha 0.60 vs 0.50 (a 20 % loss-weight bump on melanoma).
2. **MixUp + TrivialAugmentWide enabled** (E0 corrected run). Train balanced accuracy 0.86 → val 0.55 was overfitting; augmentation closed the gap.
3. **EMA at 0.999** for E0 (smoother decision boundary on minority classes than 0.9998).
4. **Per-epoch checkpoints to Drive** so a Colab disconnect at hour 5 didn't burn the run.
5. **Effective-number sampler (Cui 2019)** rather than 1/n — over-corrects less and avoids degenerate batches.

**Final composite val score = 0.7 · balanced_accuracy + 0.3 · macro_F1**, used for best-checkpoint selection; this score weights minority-class performance more than plain bal_acc would.

---

# Ensemble: soft-vote + per-class thresholds

**Soft-vote inference path:**

```
image ──▶ E0(x) ──▶ softmax ──┐                    argmax of (probs − τ_c)
                              ├──▶ 0.705·p_E0 + 0.295·p_E3 ──▶ ensemble probs ─▶ predicted class
image ──▶ E3(x) ──▶ softmax ──┘
```

**Member weights** = Dirichlet random-search optimum on **VAL macro-F1** (2,000 candidates).

**Per-class thresholds** picked on VAL macro-F1 using `pick_threshold_per_class` (mode = f1):

| Class  | MEL  | NV   | BCC  | AKIEC | BKL  | DF   | VASC | OTHER |
|--------|-----:|-----:|-----:|------:|-----:|-----:|-----:|------:|
| τ      | 0.29 | 0.22 | 0.18 | 0.26  | 0.27 | 0.57 | 0.57 | 0.26  |

**Why per-class thresholds:** the rare classes (DF, VASC) have ~5 val examples — argmax probabilities for them are noisy. Thresholding moves the operating point so a moderate DF probability beats a weak NV one. **Threshold-adjusted argmax → predicted label; raw probabilities → UI display + risk scoring.**

---

# Results — headline metrics

**TEST set, ensemble (E0 + E3, weights [0.705, 0.295]), n = 5,279, 1,000-iter bootstrap 95 % CI**

| Metric | Value (95 % CI) |
|---|---|
| 8-class **Balanced Accuracy** | **0.540 [0.509, 0.568]** |
| 8-class Macro F1              | 0.469 [0.441, 0.495]    |
| Top-1 accuracy                | 0.778                   |
| Top-2 accuracy                | 0.907                   |
| Top-3 accuracy                | 0.954                   |
| Binary-malignant **AUROC**    | **0.830**               |
| Binary-malignant AUPRC        | 0.614                   |
| Sensitivity @ 95 % specificity | 0.462                  |
| Sensitivity @ 90 % specificity | 0.572                  |
| Sensitivity @ 80 % specificity | 0.717                  |
| Expected Calibration Error    | 0.363                   |

**Read this as:** 95 % of the time the true label is among the model's top-3 predictions, and the binary "malignant vs benign" decision (MEL + BCC + AKIEC vs rest) clears a clinically usable AUROC bar.

---

# Results — per-class breakdown (TEST)

| Class  | n     | Precision           | Recall              | F1                  |
|--------|------:|---------------------|---------------------|---------------------|
| MEL    |   492 | 0.590 [0.535, 0.636] | 0.415 [0.371, 0.456] | 0.487 [0.444, 0.526] |
| NV     | 3,858 | 0.865 [0.854, 0.877] | **0.907** [0.898, 0.916] | **0.886** [0.879, 0.894] |
| BCC    |   365 | 0.699 [0.645, 0.752] | 0.471 [0.423, 0.522] | 0.563 [0.518, 0.608] |
| AKIEC  |   109 | 0.389 [0.302, 0.490] | 0.339 [0.263, 0.415] | 0.363 [0.289, 0.440] |
| BKL    |   310 | 0.482 [0.415, 0.550] | 0.339 [0.290, 0.394] | 0.398 [0.347, 0.453] |
| DF     |    57 | 0.294 [0.216, 0.368] | **0.737** [0.620, 0.841] | 0.420 [0.325, 0.499] |
| VASC   |    39 | 0.241 [0.167, 0.317] | **0.846** [0.731, 0.947] | 0.375 [0.281, 0.463] |
| OTHER  |    49 | 0.260 [0.138, 0.375] | 0.265 [0.133, 0.398] | 0.263 [0.135, 0.373] |

**Observations:** NV F1 is excellent (0.886) — the model has no trouble with the majority class. **DF and VASC recall are the surprise wins** — 0.74 and 0.85 — driven by E3's class-balanced focal loss + the per-class thresholds. The hard classes are AKIEC and OTHER (small support, fuzzy class boundary).

---

# Results — domain analysis (the smartphone tail)

| Domain         | n     | Bal. Acc.            | Macro F1             |
|----------------|------:|---------------------|---------------------|
| dermatoscopic  | 5,180 | 0.565 [0.531, 0.598] | 0.486 [0.457, 0.514] |
| **smartphone** |    99 | **0.258 [0.169, 0.379]** | **0.152 [0.060, 0.234]** |

**This is the limitation reviewers will ask about first.** 30 pp drop on smartphone is a real domain gap, not noise — the 95 % CI on smartphone (0.17 – 0.38) sits entirely below the dermatoscopic CI floor (0.53).

**Why:** PAD-UFES-20 is 2.5 % of the train data. The model learns dermatoscopic features dominantly, and smartphone images of e.g. melanoma are *out of distribution*.

**Mitigations explored / proposed:**
- **Tested:** CLIP-style image-aware class-balanced sampler boosting smartphone images per batch — improved smartphone bal_acc by ~5 pp but cost ~3 pp on dermatoscopic.
- **Future:** add an explicit smartphone-augmentation pass (random JPEG quality, motion blur, color jitter at training); or a MILK-style balanced collection.

---

# Comparison & limitations

**Vs. ISIC 2019 leaderboard (8-class balanced accuracy on the public-test split — methodologies differ but the ranking is informative):**

| Rank | System | Bal. Acc. |
|---|---|---:|
| 1 | DAISYLab (Hamburg) | 0.636 |
| 2 | DysionAI (Beijing) | 0.607 |
| 3 | AImageLab (Modena) | 0.593 |
| 5 | Nurithm Labs       | 0.569 |
| **—** | **DermaFusion (this work)** | **0.540** |
| 64 | Median challenge entry | ~0.45 |

**Where we are:** above the median, below the top-3.
**Why we're below top-3:** they ensemble 5–7 models (ResNeXt + EfficientNet + SENet154); we ship 2. The single largest accuracy lever we declined is **adding a ConvNeXt-Base member (E2)** — would lift bal_acc by ~3 pp on prior offline experiments, at the cost of doubling the iOS download size.

**Other limitations:** ECE = 0.363 (not well-calibrated; temperature scaling is the next obvious fix). Smartphone domain gap is the elephant in the room. Test set is not held-out from the official challenge (we built our own from the unified dataset) — numbers are *comparable* to the leaderboard, not directly substitutable.

---

# Deployment

**Two surfaces, one model.**

**Gradio research demo** — `python -m demo.app` →
- Loads `dermafusion_ensemble.pt` (single 142 MB pickled module)
- Shows top-K probabilities, GradCAM heatmap, MEL-decision threshold readout
- Currently running locally at http://127.0.0.1:55827

**iOS app (DermFusion)** —
- Two `.mlpackage` files (FP16, 34 MB each) bundled via Xcode synchronized folders
- `ModelService` actor: loads both models, runs them concurrently with `async let`, weighted-averages in Swift, applies per-class thresholds for the primary diagnosis label
- Image preprocessing (Shades-of-Gray + DullRazor + center-crop) reproduced in Swift to match training distribution
- Quality gates (resolution, exposure) and uncertainty gates (top-1 < 0.15, near-uniform entropy) layered on top of the model output
- Verified PyTorch ↔ CoreML output diff: max |Δ| = 0.007 on a single B4 (FP16 quantization noise)

**What's next:** ConvNeXt-Base member (~+3 pp bal_acc), temperature scaling for calibration (~ECE 0.36 → 0.10), explicit smartphone-augmentation training pass to close the 30 pp domain gap, and a **clinician-in-the-loop** UI flow that surfaces top-3 candidates rather than a single prediction when ensemble confidence is low.

---
