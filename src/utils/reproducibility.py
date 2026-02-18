"""Reproducibility and environment utilities."""

from __future__ import annotations

import logging
import platform
import random

import numpy as np
import torch

__all__ = ["get_device", "log_system_info", "seed_everything"]


def seed_everything(seed: int = 42) -> None:
    """Seed all random number generators for reproducible runs.

    Args:
        seed: Global random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(override: str | None = None) -> torch.device:
    """Return the best available compute device, or the override if set.

    Args:
        override: One of "cuda", "mps", "cpu", or None for auto. Use "cpu" on
            macOS if MPS hangs on first batch (e.g. first tensor transfer or forward).
    """
    if override is not None and str(override).strip().lower() not in ("none", "null", ""):
        s = str(override).strip().lower()
        if s == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if s == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if s in ("cpu", "cuda", "mps"):
            return torch.device(s)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def log_system_info(logger: logging.Logger) -> None:
    """Log Python and PyTorch runtime details.

    Args:
        logger: Logger used to emit system information.
    """
    logger.info("Python version: %s", platform.python_version())
    logger.info("PyTorch version: %s", torch.__version__)
    logger.info("CUDA available: %s", torch.cuda.is_available())
    if torch.cuda.is_available():
        logger.info("CUDA version: %s", torch.version.cuda)
        logger.info("GPU: %s", torch.cuda.get_device_name(0))
        memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info("GPU memory: %.2f GB", memory_gb)

