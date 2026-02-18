"""Step 5 tests for Trainer behavior."""

from __future__ import annotations

from pathlib import Path
import tempfile

from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.models.model_factory import build_model
from src.training.callbacks import ModelCheckpoint
from src.training.losses import FocalLoss
from src.training.optimizer import build_optimizer
from src.training.trainer import Trainer


class TinyDictDataset(Dataset):
    """Tiny dict-based dataset for trainer tests."""

    def __init__(self, n: int = 16) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        label = idx % 7
        return {
            "image": torch.randn(3, 64, 64),
            "metadata": torch.tensor([0.5, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


class TinyFusionModel(nn.Module):
    """Small multimodal model for fast trainer tests."""

    def __init__(self, fusion_type: str = "late") -> None:
        super().__init__()
        self.fusion_type = fusion_type
        self.image_proj = nn.Sequential(nn.Flatten(), nn.Linear(3 * 64 * 64, 64), nn.ReLU())
        self.meta_proj = nn.Linear(6, 64)
        self.head = nn.Linear(64, 7)

    def forward(self, batch):
        img = self.image_proj(batch["image"])
        meta = self.meta_proj(batch["metadata"])
        if self.fusion_type == "late":
            fused = 0.5 * img + 0.5 * meta
        elif self.fusion_type == "film":
            fused = img * torch.sigmoid(meta)
        else:
            fused = img + meta
        return self.head(fused)


def _trainer_config(epochs: int = 2, mixed_precision: bool = False):
    return OmegaConf.create(
        {
            "training": {
                "training": {
                    "epochs": epochs,
                    "gradient_clip_val": 1.0,
                    "accumulate_grad_batches": 1,
                    "mixed_precision": mixed_precision,
                    "scheduler": {"warmup_epochs": 1},
                    "optimizer": {"lr": 1e-3, "head_lr": 1e-3, "weight_decay": 0.0},
                }
            }
        }
    )


def test_trainer_runs_two_epochs() -> None:
    """Trainer should run fit for two epochs on tiny data."""
    cfg = _trainer_config(epochs=2)
    dataset = TinyDictDataset(12)
    loader = DataLoader(dataset, batch_size=4)
    model = TinyFusionModel("late")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=cfg,
        device=torch.device("cpu"),
        loss_fn=FocalLoss(),
        optimizer=optimizer,
    )
    results = trainer.fit()
    assert results["epochs_ran"] == 2


def test_checkpointing_saves_and_loads() -> None:
    """Trainer checkpoint callback should save reusable files."""
    cfg = _trainer_config(epochs=1)
    dataset = TinyDictDataset(8)
    loader = DataLoader(dataset, batch_size=4)
    model = TinyFusionModel("late")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmp:
        checkpoint = ModelCheckpoint(save_dir=Path(tmp))
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=cfg,
            device=torch.device("cpu"),
            loss_fn=FocalLoss(),
            optimizer=optimizer,
            checkpoint=checkpoint,
        )
        trainer.fit()
        last_path = Path(tmp) / "last.ckpt"
        assert last_path.exists()
        payload = torch.load(last_path, map_location="cpu")
        reloaded = TinyFusionModel("late")
        reloaded.load_state_dict(payload["model_state"])


def test_training_completes_for_all_fusion_modes() -> None:
    """Trainer should complete one epoch for late/film/cross-attention style flows."""
    dataset = TinyDictDataset(10)
    loader = DataLoader(dataset, batch_size=5)
    cfg = _trainer_config(epochs=1)
    for fusion in ("late", "film", "cross_attention"):
        model = TinyFusionModel(fusion)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=cfg,
            device=torch.device("cpu"),
            loss_fn=FocalLoss(),
            optimizer=optimizer,
        )
        result = trainer.fit()
        assert result["epochs_ran"] >= 1


def test_mixed_precision_flag_and_gradient_clipping() -> None:
    """Trainer should run with mixed_precision flag and invoke grad clipping."""
    cfg = _trainer_config(epochs=1, mixed_precision=True)
    dataset = TinyDictDataset(8)
    loader = DataLoader(dataset, batch_size=4)
    model = TinyFusionModel("late")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    called = {"count": 0}
    original = torch.nn.utils.clip_grad_norm_

    def _patched(*args, **kwargs):
        called["count"] += 1
        return original(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = _patched
    try:
        trainer = Trainer(
            model=model,
            train_loader=loader,
            val_loader=loader,
            config=cfg,
            device=torch.device("cpu"),
            loss_fn=FocalLoss(),
            optimizer=optimizer,
        )
        trainer.fit()
    finally:
        torch.nn.utils.clip_grad_norm_ = original

    assert called["count"] > 0


def test_build_model_instantiation_for_train_script_contract() -> None:
    """Factory-produced model should be compatible with trainer batch contract."""
    cfg = OmegaConf.create(
        {
            "model": {
                "model": {
                    "backbone": "efficientnet_b4",
                    "pretrained": False,
                    "use_metadata": False,
                    "num_classes": 7,
                }
            }
        }
    )
    model = build_model(cfg)
    batch = {"image": torch.randn(1, 3, 224, 224)}
    logits = model(batch)
    assert logits.shape == (1, 7)
