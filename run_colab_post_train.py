"""Colab-friendly launcher for scripts/colab_post_training.py.

Usage from Colab:
  python run_colab_post_train.py --checkpoint /content/DermaFusion/outputs/checkpoints/best.ckpt ...
  python run_colab_post_train.py --find-checkpoint-in-drive --find-isic-in-drive ...
"""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

runpy.run_path(str(ROOT / "scripts" / "colab_post_training.py"), run_name="__main__")
