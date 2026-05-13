"""
Colab-friendly entrypoint: set path then run training script.
Usage from Colab (project root = /content/DermaFusion):
  python run_colab_train.py data.data.use_preprocessed=true ...
"""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# Project root = directory containing this file
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

# Run the real training script (argv already set by caller)
runpy.run_path(str(ROOT / "scripts" / "train.py"), run_name="__main__")
