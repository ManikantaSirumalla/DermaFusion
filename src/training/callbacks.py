"""Training callbacks: early stopping, checkpointing, and WandB logging."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

__all__ = ["EarlyStopping", "ModelCheckpoint", "WandBCallback"]


class EarlyStopping:
    """Stop training after patience epochs without metric improvement."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 0.0) -> None:
        """Initialize early-stopping criteria."""
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best_score = float("-inf") if mode == "max" else float("inf")
        self.bad_epochs = 0

    def step(self, score: float) -> bool:
        """Update state and return True when training should stop."""
        if self.mode == "max":
            improved = score > (self.best_score + self.min_delta)
        else:
            improved = score < (self.best_score - self.min_delta)

        if improved:
            self.best_score = score
            self.bad_epochs = 0
            return False

        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


class ModelCheckpoint:
    """Save best and last checkpoints based on monitored metric."""

    def __init__(self, save_dir: Path, monitor: str = "balanced_accuracy", mode: str = "max") -> None:
        """Initialize checkpoint manager."""
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self.best_metric = float("-inf") if mode == "max" else float("inf")

    def _is_better(self, metric: float) -> bool:
        return metric > self.best_metric if self.mode == "max" else metric < self.best_metric

    def save(
        self,
        epoch: int,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
        metrics: dict[str, float],
        config: Any | None = None,
    ) -> None:
        """Save latest checkpoint and best checkpoint if improved."""
        payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": None if optimizer is None else optimizer.state_dict(),
            "scheduler_state": None if scheduler is None else getattr(scheduler, "__dict__", None),
            "metrics": metrics,
            "config": config,
        }
        torch.save(payload, self.save_dir / "last.ckpt")

        metric = float(metrics.get(self.monitor, 0.0))
        if self._is_better(metric):
            self.best_metric = metric
            torch.save(payload, self.save_dir / "best.ckpt")


class WandBCallback:
    """Thin adapter for optional WandB logging."""

    def __init__(self, run: Any | None = None) -> None:
        """Store run handle; if None, logging is skipped."""
        self.run = run

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to WandB when available."""
        if self.run is None:
            return
        self.run.log(metrics, step=step)
