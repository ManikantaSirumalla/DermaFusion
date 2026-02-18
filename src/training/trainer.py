"""Trainer implementation with two-stage strategy and mixed-precision support."""

from __future__ import annotations

import logging
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.training.callbacks import EarlyStopping, ModelCheckpoint, WandBCallback

__all__ = ["Trainer", "TrainerState"]


@dataclass
class TrainerState:
    """Container for trainer runtime state."""

    epoch: int = 0
    best_metric: float = 0.0


def _cfg_value(cfg: Any, path: str, default: Any) -> Any:
    """Access nested config path for dict/dataclass/omegaconf objects."""
    current = cfg
    for key in path.split("."):
        if isinstance(current, dict):
            current = current.get(key, default)
        else:
            current = getattr(current, key, default)
    return current


class Trainer:
    """Complete training pipeline with optional two-stage strategy."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Any,
        device: torch.device,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any | None = None,
        stage2_loss_fn: nn.Module | None = None,
        early_stopping: EarlyStopping | None = None,
        checkpoint: ModelCheckpoint | None = None,
        wandb_cb: WandBCallback | None = None,
    ) -> None:
        """Initialize trainer dependencies and runtime state."""
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        self.loss_fn = loss_fn
        self.stage2_loss_fn = stage2_loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.early_stopping = early_stopping
        self.checkpoint = checkpoint
        self.wandb_cb = wandb_cb or WandBCallback(run=None)
        self.state = TrainerState()

        self.grad_clip_val = float(_cfg_value(config, "training.training.gradient_clip_val", 1.0))
        self.accumulate_steps = int(
            _cfg_value(config, "training.training.accumulate_grad_batches", 1)
        )
        self.max_epochs = int(_cfg_value(config, "training.training.epochs", 1))
        self.warmup_epochs = int(_cfg_value(config, "training.training.scheduler.warmup_epochs", 0))
        self.use_mixed_precision = bool(_cfg_value(config, "training.training.mixed_precision", False))
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_mixed_precision and device.type == "cuda")

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Move tensor items in batch dict to trainer device."""
        output: dict[str, Any] = {}
        for key, value in batch.items():
            output[key] = value.to(self.device) if torch.is_tensor(value) else value
        return output

    def _compute_metrics(self, logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
        """Compute balanced accuracy and macro-F1 for a mini-batch aggregate."""
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        true = labels.detach().cpu().numpy()
        return {
            "balanced_accuracy": float(balanced_accuracy_score(true, preds)),
            "macro_f1": float(f1_score(true, preds, average="macro")),
        }

    def train_epoch(self, epoch: int | None = None) -> dict[str, float]:
        """Run one training epoch and return aggregate metrics."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        active_loss = self.loss_fn
        if (
            self.stage2_loss_fn is not None
            and epoch is not None
            and epoch >= max(self.warmup_epochs, int(self.max_epochs * 0.7))
        ):
            active_loss = self.stage2_loss_fn

        # Use autocast only on CUDA (CPU autocast can hang or be very slow on some setups)
        autocast_ctx = (
            torch.autocast(device_type="cuda", enabled=self.scaler.is_enabled())
            if self.device.type == "cuda"
            else nullcontext()
        )

        for step, batch in enumerate(self.train_loader, start=1):
            if step == 1:
                logging.getLogger("train").info("First batch received from loader.")
            batch = self._move_batch_to_device(batch)
            if step == 1:
                logging.getLogger("train").info("First batch on device, running forward...")
            labels = batch["label"].long()

            with autocast_ctx:
                logits = self.model(batch)
                if step == 1:
                    logging.getLogger("train").info("First forward done.")
                loss = active_loss(logits, labels) / self.accumulate_steps
                if step == 1:
                    logging.getLogger("train").info("First loss computed, running backward...")

            if self.scaler.is_enabled():
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            if step == 1:
                logging.getLogger("train").info("First backward done.")

            if step % self.accumulate_steps == 0:
                if self.scaler.is_enabled():
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
                if self.scaler.is_enabled():
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if step == 1:
                    logging.getLogger("train").info("First optimizer step done.")

            total_loss += float(loss.item()) * self.accumulate_steps
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        merged_logits = torch.cat(all_logits, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)
        metrics = self._compute_metrics(merged_logits, merged_labels)
        metrics["loss"] = total_loss / max(1, len(self.train_loader))
        return metrics

    @torch.no_grad()
    def validate(self) -> dict[str, float]:
        """Run validation pass and return aggregate metrics."""
        self.model.eval()
        total_loss = 0.0
        all_logits: list[torch.Tensor] = []
        all_labels: list[torch.Tensor] = []

        for batch in self.val_loader:
            batch = self._move_batch_to_device(batch)
            labels = batch["label"].long()
            logits = self.model(batch)
            loss = self.loss_fn(logits, labels)
            total_loss += float(loss.item())
            all_logits.append(logits.detach().cpu())
            all_labels.append(labels.detach().cpu())

        merged_logits = torch.cat(all_logits, dim=0)
        merged_labels = torch.cat(all_labels, dim=0)
        metrics = self._compute_metrics(merged_logits, merged_labels)
        metrics["loss"] = total_loss / max(1, len(self.val_loader))
        return metrics

    def _log_epoch(self, epoch: int, train_metrics: dict[str, float], val_metrics: dict[str, float]) -> None:
        """Log epoch metrics to optional WandB callback."""
        metrics = {f"train/{k}": v for k, v in train_metrics.items()}
        metrics.update({f"val/{k}": v for k, v in val_metrics.items()})
        metrics["epoch"] = epoch
        self.wandb_cb.log_metrics(metrics, step=epoch)

    def fit(self) -> dict[str, float]:
        """Run full training loop with optional callbacks."""
        best_val = float("-inf")
        for epoch in range(self.max_epochs):
            self.state.epoch = epoch
            if epoch == 0:
                logging.getLogger("train").info("Epoch 1/%d: loading first batch...", self.max_epochs)
            train_metrics = self.train_epoch(epoch=epoch)
            val_metrics = self.validate()

            if self.scheduler is not None and hasattr(self.scheduler, "step"):
                self.scheduler.step(epoch)

            self._log_epoch(epoch, train_metrics, val_metrics)

            current_val = float(val_metrics.get("balanced_accuracy", 0.0))
            if current_val > best_val:
                best_val = current_val
                self.state.best_metric = best_val

            if self.checkpoint is not None:
                self.checkpoint.save(
                    epoch=epoch,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    metrics=val_metrics,
                    config=self.config,
                )

            if self.early_stopping is not None and self.early_stopping.step(current_val):
                break

        return {"best_balanced_accuracy": self.state.best_metric, "epochs_ran": self.state.epoch + 1}

    def run_sanity_check(self, iterations: int = 50) -> tuple[float, float]:
        """Overfit one batch; useful to quickly validate optimization path."""
        self.model.train()
        batch = next(iter(self.train_loader))
        batch = self._move_batch_to_device(batch)
        labels = batch["label"].long()

        start_loss = float(self.loss_fn(self.model(batch), labels).item())
        for _ in range(iterations):
            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(batch)
            loss = self.loss_fn(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_val)
            self.optimizer.step()
        end_loss = float(self.loss_fn(self.model(batch), labels).item())
        return start_loss, end_loss
