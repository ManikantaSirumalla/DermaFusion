#!/usr/bin/env bash
# Run baseline training from project root or scripts/.
# Usage:
#   ./scripts/run_training.sh                    # full training (50 epochs)
#   ./scripts/run_training.sh sanity            # sanity check only (overfit 1 batch)
#   ./scripts/run_training.sh multimodality     # late fusion with metadata
#
# Optional env: FORCE_CPU=1 to use CPU only (e.g. if MPS/CUDA crashes with exit 139).

set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VENV_PYTHON="${VENV_PYTHON:-$REPO_ROOT/.venv311/bin/python}"
if [[ ! -x "$VENV_PYTHON" ]]; then
  echo "Expected venv Python at $VENV_PYTHON. Set VENV_PYTHON or create .venv311."
  exit 1
fi

if [[ "${FORCE_CPU}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

case "${1:-}" in
  sanity)
    echo "Running sanity check (overfit 1 batch)..."
    "$VENV_PYTHON" scripts/train.py training.training.sanity_check=true
    ;;
  multimodality)
    echo "Running late-fusion multimodal training..."
    "$VENV_PYTHON" scripts/train.py experiment=multimodal_late model.use_metadata=true experiment_name=late_fusion_efficientnet
    ;;
  *)
    echo "Running baseline training (EfficientNet image-only, 50 epochs)..."
    "$VENV_PYTHON" scripts/train.py
    ;;
esac
