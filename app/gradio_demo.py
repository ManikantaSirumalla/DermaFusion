"""Interactive Gradio demo (guide: app/gradio_demo.py).

Runs the same demo as demo/app.py. Launch with:
  python app/gradio_demo.py
  or
  python -m app.gradio_demo
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if __name__ == "__main__":
    from demo.app import demo
    demo.launch()
