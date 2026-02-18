"""Logging and experiment tracking helpers."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

__all__ = ["setup_logger", "setup_wandb"]


def setup_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Create and configure a logger with console and optional file handlers.

    Args:
        name: Logger name.
        level: Logging verbosity level.
        log_file: Optional file path for persisted logs.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_wandb(
    config: dict[str, Any],
    project_name: str,
    run_name: str,
):
    """Initialize Weights & Biases with offline fallback.

    Args:
        config: Resolved experiment config dictionary.
        project_name: W&B project name.
        run_name: W&B run name.

    Returns:
        A W&B run object.
    """
    try:
        import wandb  # Local import to avoid hard dependency for non-training scripts.
    except Exception:
        return None

    # Use offline mode if WANDB_MODE=offline or if user is not logged in (avoids prompt).
    import os
    mode = os.environ.get("WANDB_MODE", "offline")
    return wandb.init(
        project=project_name,
        name=run_name,
        config=config,
        mode=mode,
    )

