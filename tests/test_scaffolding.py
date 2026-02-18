"""Step 1 scaffolding tests."""

from __future__ import annotations

from pathlib import Path


def test_project_directories_exist() -> None:
    """Required Step 1 directories should be present."""
    required_dirs = [
        "configs",
        "configs/model",
        "configs/training",
        "configs/data",
        "configs/experiment",
        "src",
        "src/data",
        "src/models",
        "src/training",
        "src/evaluation",
        "src/utils",
        "scripts",
        "tests",
        "demo",
    ]
    for directory in required_dirs:
        assert Path(directory).is_dir(), f"Missing directory: {directory}"

