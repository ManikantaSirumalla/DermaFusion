"""Learning-rate scheduling utilities."""

from __future__ import annotations

import math

import torch

__all__ = ["CosineAnnealingWithWarmup", "CosineAnnealingWarmRestarts"]


class CosineAnnealingWarmRestarts:
    """Cosine annealing with warm restarts (guide: T_0=10, T_mult=2)."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        T_0: int = 10,
        T_mult: int = 2,
        eta_min: float = 1e-6,
    ) -> None:
        self.optimizer = optimizer
        self.T_0 = T_0
        self.T_mult = T_mult
        self.eta_min = eta_min
        self.base_lrs = [g["lr"] for g in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update LRs for current epoch (0-indexed)."""
        T_cur = epoch
        T_i = self.T_0
        cycle = 0
        while T_cur >= T_i:
            T_cur -= T_i
            T_i *= self.T_mult
            cycle += 1
        progress = T_cur / T_i
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            group["lr"] = self.eta_min + (base_lr - self.eta_min) * cosine


class CosineAnnealingWithWarmup:
    """Cosine annealing scheduler with linear warmup."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-6,
    ) -> None:
        """Store scheduler parameters."""
        self.optimizer = optimizer
        self.warmup_epochs = max(1, warmup_epochs)
        self.total_epochs = max(self.warmup_epochs + 1, total_epochs)
        self.min_lr = min_lr
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def step(self, epoch: int) -> None:
        """Update optimizer learning rates for current epoch index."""
        for i, group in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if epoch < self.warmup_epochs:
                lr = base_lr * float(epoch + 1) / float(self.warmup_epochs)
            else:
                progress = (epoch - self.warmup_epochs) / float(
                    self.total_epochs - self.warmup_epochs
                )
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                lr = self.min_lr + (base_lr - self.min_lr) * cosine
            group["lr"] = lr
