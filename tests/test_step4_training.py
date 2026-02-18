"""Step 4 tests for losses, optimizer, scheduler, and callbacks."""

from __future__ import annotations

from pathlib import Path
import tempfile

import torch
from omegaconf import OmegaConf

from src.models.model_factory import build_model
from src.training.callbacks import EarlyStopping, ModelCheckpoint
from src.training.losses import CostSensitiveLoss, FocalLoss
from src.training.optimizer import build_optimizer
from src.training.schedulers import CosineAnnealingWithWarmup


def test_focal_loss_returns_positive_value() -> None:
    """Focal loss should produce finite positive scalar."""
    loss_fn = FocalLoss(gamma=2.0)
    logits = torch.tensor([[2.0, 0.1, -0.2], [0.3, 1.2, -0.4]], dtype=torch.float32)
    targets = torch.tensor([0, 1], dtype=torch.long)
    loss = loss_fn(logits, targets)
    assert torch.isfinite(loss)
    assert float(loss.item()) > 0.0


def test_cost_sensitive_loss_respects_class_weights() -> None:
    """Loss object should register higher melanoma class weight."""
    cost_matrix = {"mel": {"fn_weight": 5.0}, "default": {"fn_weight": 1.0}}
    loss_fn = CostSensitiveLoss(cost_matrix=cost_matrix, num_classes=7)
    assert float(loss_fn.class_weights[0]) == 5.0
    assert float(loss_fn.class_weights[1]) == 1.0


def test_optimizer_param_groups_use_different_lrs() -> None:
    """Optimizer should set lower LR for backbone and higher LR for heads."""
    cfg = OmegaConf.create(
        {
            "training": {
                "training": {"optimizer": {"lr": 1e-4, "head_lr": 1e-3, "weight_decay": 0.01}}
            },
            "model": {"backbone": "efficientnet_b4", "pretrained": False, "use_metadata": False},
        }
    )
    model = build_model(cfg)
    optimizer = build_optimizer(model, cfg)
    lrs = sorted({group["lr"] for group in optimizer.param_groups})
    assert lrs == [1e-4, 1e-3]


def test_scheduler_warmup_then_decay() -> None:
    """Scheduler should increase during warmup then decay afterwards."""
    param = torch.nn.Parameter(torch.randn(1))
    optimizer = torch.optim.AdamW([param], lr=1e-3)
    scheduler = CosineAnnealingWithWarmup(
        optimizer=optimizer,
        warmup_epochs=2,
        total_epochs=10,
        min_lr=1e-6,
    )
    scheduler.step(0)
    lr0 = optimizer.param_groups[0]["lr"]
    scheduler.step(1)
    lr1 = optimizer.param_groups[0]["lr"]
    scheduler.step(9)
    lr_last = optimizer.param_groups[0]["lr"]
    assert lr1 >= lr0
    assert lr_last < lr1


def test_early_stopping_patience_trigger() -> None:
    """Early stopping should trigger after patience epochs without improvement."""
    stopper = EarlyStopping(patience=2, mode="max")
    assert stopper.step(0.5) is False
    assert stopper.step(0.4) is False
    assert stopper.step(0.3) is True


def test_model_checkpoint_saves_best_and_last() -> None:
    """Checkpoint callback should write last and best checkpoint files."""
    cfg = OmegaConf.create({"model": {"backbone": "efficientnet_b4", "pretrained": False}})
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with tempfile.TemporaryDirectory() as tmp_dir:
        callback = ModelCheckpoint(save_dir=Path(tmp_dir), monitor="balanced_accuracy", mode="max")
        callback.save(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            metrics={"balanced_accuracy": 0.6},
            config=None,
        )
        assert (Path(tmp_dir) / "last.ckpt").exists()
        assert (Path(tmp_dir) / "best.ckpt").exists()
