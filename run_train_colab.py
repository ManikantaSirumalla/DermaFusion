#!/usr/bin/env python3
"""Launcher for Colab: adds project root to path then runs scripts/train.py with same argv."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_COLAB = Path("/content/DermaFusion")
for _d in (_ROOT, _COLAB):
    if _d.exists():
        _s = str(_d)
        if _s not in sys.path:
            sys.path.insert(0, _s)

# Force "src" to be loaded in this process so train.py's imports see it
import src.data.dataset  # noqa: F401

# Run train.py with current argv (keep all override args)
_train_script = _ROOT / "scripts" / "train.py"
if not _train_script.exists():
    _train_script = _COLAB / "scripts" / "train.py"
sys.argv[0] = str(_train_script)
runpy.run_path(str(_train_script), run_name="__main__")
