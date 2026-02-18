"""Data download & preprocessing entry point (guide: scripts/prepare_data.py).

Same CLI as setup_data.py. Usage:
  python scripts/prepare_data.py --raw-data-dir /path/to/ISIC2018_Task3 --output-dir data/raw
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    runpy.run_path(str(PROJECT_ROOT / "scripts" / "setup_data.py"), run_name="__main__")
