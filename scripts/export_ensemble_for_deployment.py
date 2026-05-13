"""Build the E0+E3 ensemble deployment artifacts (PyTorch bundle + CoreML mlpackage).

Run this in Colab where the trained checkpoints already live on Drive. It
produces two files in --out-dir:

  dermafusion_ensemble_bundle.pt   # for Gradio (loaded by inference_runtime.py)
  DermaFusion_B4.mlpackage         # for iOS (drop-in replacement)

Defaults match the run paths recorded by Phase 9.6 of the notebook.

Usage (Colab):
  !cd /content/DermaFusion && python scripts/export_ensemble_for_deployment.py \
      --out-dir /content/drive/MyDrive/DermaFusion_archive/deploy_v_final
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import torch


CLASS_TO_IDX = {
    "MEL": 0, "NV": 1, "BCC": 2, "AKIEC": 3,
    "BKL": 4, "DF": 5, "VASC": 6, "OTHER": 7,
}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 380
DEFAULT_MEL_THRESHOLD = 0.30


@dataclass
class MemberCfg:
    name: str
    run_dir: Path
    backbone: str
    prefer_kind: str  # "raw" or "ema"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--archive-root",
        default="/content/drive/MyDrive/DermaFusion_archive",
        help="Drive folder containing the per-run subfolders.",
    )
    p.add_argument(
        "--e0-dir",
        default=None,
        help="Override E0 run dir (defaults to archive-root/E0corr_b4_20260508_170853).",
    )
    p.add_argument(
        "--e3-dir",
        default=None,
        help="Override E3 run dir (defaults to archive-root/E3effb4_focal_20260508_215741).",
    )
    p.add_argument(
        "--ensemble-json",
        default=None,
        help="Path to final_ensemble.json for member weights. "
             "Defaults to archive-root/ensemble_E0_E3/final_ensemble.json.",
    )
    p.add_argument(
        "--out-dir",
        required=True,
        help="Where to write the .pt bundle and .mlpackage.",
    )
    p.add_argument(
        "--coreml-precision",
        choices=["float16", "float32"],
        default="float16",
        help="CoreML weight precision. float16 halves disk size with negligible accuracy loss.",
    )
    p.add_argument(
        "--ios-target",
        default="iOS15",
        help="Minimum CoreML deployment target (e.g. iOS15, iOS16, iOS17).",
    )
    p.add_argument(
        "--skip-coreml",
        action="store_true",
        help="Build only the .pt bundle; skip CoreML conversion.",
    )
    return p.parse_args()


def _load_member_state_dict(run_dir: Path, prefer_kind: str) -> tuple[dict, str, float]:
    """Pick best.ckpt vs best_ema.ckpt by best_val_score; return (state_dict, kind, score)."""
    candidates: list[tuple[str, Path]] = []
    if prefer_kind == "ema":
        candidates += [("ema", run_dir / "best_ema.ckpt"), ("raw", run_dir / "best.ckpt")]
    else:
        candidates += [("raw", run_dir / "best.ckpt"), ("ema", run_dir / "best_ema.ckpt")]

    scored: list[tuple[float, str, Path, dict]] = []
    for kind, path in candidates:
        if not path.exists():
            continue
        ck = torch.load(path, map_location="cpu", weights_only=False)
        score = float(ck.get("best_val_score", ck.get("val_score", -1.0)))
        sd = ck.get("state_dict", ck.get("model_state_dict", ck))
        if not isinstance(sd, dict):
            continue
        # Strip common prefixes some training scripts add.
        sd = {k.replace("module.", "").replace("model.", "", 1) if k.startswith("model.") else k.replace("module.", ""): v for k, v in sd.items()}
        scored.append((score, kind, path, sd))

    if not scored:
        raise FileNotFoundError(f"No best/best_ema ckpt under {run_dir}")
    scored.sort(key=lambda t: t[0], reverse=True)
    score, kind, path, sd = scored[0]
    print(f"  [member] {run_dir.name}: chose {kind} ({path.name}) score={score:.4f}")
    return sd, kind, score


def _load_weights(ensemble_json: Path | None, default_n: int) -> list[float]:
    if ensemble_json is not None and ensemble_json.exists():
        try:
            data = json.loads(ensemble_json.read_text())
            w = data.get("weights")
            if isinstance(w, list) and len(w) >= default_n:
                ws = [float(x) for x in w[:default_n]]
                s = sum(ws) or 1.0
                return [x / s for x in ws]
        except Exception as e:
            print(f"  [warn] could not parse {ensemble_json}: {e}")
    return [1.0 / default_n] * default_n


def main() -> int:
    args = _parse_args()

    # Make `src.*` importable when run from project root.
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    from src.deployment.ensemble import EnsembleSoftVote, MemberSpec

    archive = Path(args.archive_root)
    e0_dir = Path(args.e0_dir) if args.e0_dir else archive / "E0corr_b4_20260508_170853"
    e3_dir = Path(args.e3_dir) if args.e3_dir else archive / "E3effb4_focal_20260508_215741"
    ensemble_json = (
        Path(args.ensemble_json)
        if args.ensemble_json
        else archive / "ensemble_E0_E3" / "final_ensemble.json"
    )
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    members_cfg = [
        MemberCfg("E0_baseline_b4", e0_dir, "efficientnet_b4", "raw"),
        MemberCfg("E3_focal_balanced_b4", e3_dir, "tf_efficientnet_b4.ns_jft_in1k", "raw"),
    ]
    print("Loading checkpoints:")
    member_states: list[dict] = []
    for m in members_cfg:
        sd, _, _ = _load_member_state_dict(m.run_dir, m.prefer_kind)
        member_states.append(sd)

    weights = _load_weights(ensemble_json, len(members_cfg))
    print(f"Member weights: {weights}")

    specs = [MemberSpec(backbone=m.backbone, weight=w) for m, w in zip(members_cfg, weights)]
    ensemble = EnsembleSoftVote(specs, num_classes=len(CLASS_TO_IDX))
    ensemble.eval()

    # Load each submodel's state_dict.
    for i, sd in enumerate(member_states):
        missing, unexpected = ensemble.members[i].load_state_dict(sd, strict=False)
        if missing or unexpected:
            print(f"  [member {i}] {len(missing)} missing, {len(unexpected)} unexpected keys "
                  f"(first missing: {missing[:3]})")

    # ---------------- 1) PyTorch bundle for Gradio ----------------
    bundle_payload = {
        "model_loader": "ensemble",
        "model_name": "ensemble_b4_x2",
        "members": [{"backbone": s.backbone, "weight": float(w)} for s, w in zip(specs, weights)],
        "state_dict": ensemble.state_dict(),
        "class_to_idx": CLASS_TO_IDX,
        "num_classes": len(CLASS_TO_IDX),
        "img_size": IMG_SIZE,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "mel_threshold": DEFAULT_MEL_THRESHOLD,
        "use_metadata": False,
    }
    bundle_path = out_dir / "dermafusion_ensemble_bundle.pt"
    torch.save(bundle_payload, bundle_path)
    print(f"\nWrote {bundle_path}  ({bundle_path.stat().st_size / 1e6:.1f} MB)")

    # ---------------- 2) CoreML mlpackage for iOS ----------------
    if args.skip_coreml:
        print("Skipping CoreML conversion (--skip-coreml).")
        return 0

    try:
        import coremltools as ct
    except ImportError:
        print("coremltools not installed. In Colab: !pip install -q coremltools")
        return 1

    # Move ensemble to CPU + float32 for tracing.
    ensemble = ensemble.to("cpu").float().eval()
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)
    print("\nTracing ensemble (this can take ~30s) ...")
    with torch.no_grad():
        traced = torch.jit.trace(ensemble, example, strict=False)

    target_map = {
        "iOS15": ct.target.iOS15,
        "iOS16": ct.target.iOS16,
        "iOS17": ct.target.iOS17,
    }
    target = target_map.get(args.ios_target, ct.target.iOS15)
    precision = (
        ct.precision.FLOAT16 if args.coreml_precision == "float16" else ct.precision.FLOAT32
    )

    print(f"Converting to CoreML (target={args.ios_target}, precision={args.coreml_precision}) ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input_image", shape=(1, 3, IMG_SIZE, IMG_SIZE))],
        outputs=[ct.TensorType(name="logits")],
        convert_to="mlprogram",
        compute_precision=precision,
        minimum_deployment_target=target,
    )
    mlmodel.short_description = (
        "DermaFusion E0+E3 soft-vote ensemble (EfficientNet-B4 x2, 380px). "
        "Output 'logits' = log(weighted_avg_softmax). "
        "softmax(logits) recovers the ensemble probabilities exactly."
    )
    mlmodel.author = "DermaFusion"

    mlpackage_path = out_dir / "DermaFusion_B4.mlpackage"
    mlmodel.save(str(mlpackage_path))

    weights_bin = mlpackage_path / "Data" / "com.apple.CoreML" / "weights" / "weight.bin"
    sz = weights_bin.stat().st_size / 1e6 if weights_bin.exists() else 0.0
    print(f"Wrote {mlpackage_path}  (weights ~{sz:.1f} MB)")

    # Sanity check: feed example through both PyTorch and CoreML, compare.
    print("\nVerifying CoreML vs PyTorch on a random input ...")
    with torch.no_grad():
        pt_out = ensemble(example).numpy()
    cm_out = mlmodel.predict({"input_image": example.numpy()})["logits"]
    diff = float(abs(pt_out - cm_out).max())
    print(f"  max |PyTorch - CoreML| logit diff: {diff:.4e}")
    if diff > 5e-2:
        print("  WARNING: diff is large; CoreML graph may have lossy fp16 ops.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
