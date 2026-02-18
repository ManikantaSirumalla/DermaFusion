"""Step 1 reproducibility tests."""

from __future__ import annotations

import random

import numpy as np
import torch

from src.utils.reproducibility import seed_everything


def test_seed_everything_reproducible() -> None:
    """Repeated seeding should reproduce random values."""
    seed_everything(123)
    python_first = random.random()
    numpy_first = np.random.rand()
    torch_first = torch.rand(1).item()

    seed_everything(123)
    python_second = random.random()
    numpy_second = np.random.rand()
    torch_second = torch.rand(1).item()

    assert python_first == python_second
    assert numpy_first == numpy_second
    assert torch_first == torch_second

