"""Hydra entrypoint for model training."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import HAM10000Dataset
from src.data.preprocessing import encode_metadata, compute_class_weights
from src.data.sampler import ClassBalancedSampler
from src.data.transforms import get_train_transforms, get_val_transforms
from src.models.model_factory import build_model
from src.training.callbacks import EarlyStopping, ModelCheckpoint, WandBCallback
from src.training.losses import CostSensitiveLoss, FocalLoss
from src.training.optimizer import build_optimizer
from src.training.schedulers import CosineAnnealingWithWarmup
from src.training.trainer import Trainer
from src.utils.config import validate_config
from src.utils.logging import setup_logger, setup_wandb
from src.utils.reproducibility import get_device, seed_everything

CLASS_ORDER = ["mel", "nv", "bcc", "akiec", "bkl", "df", "vasc"]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_ORDER)}
GT_CLASS_COLS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def _resolve_path(path_value: str) -> Path:
    """Resolve config path to absolute project path."""
    path = Path(path_value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _cfg_value(cfg: DictConfig, path: str, default: Any) -> Any:
    """Read nested config value with fallback default."""
    value = OmegaConf.select(cfg, path, default=None)
    return default if value is None else value


def _load_aux_metadata(metadata_csv: Path) -> pd.DataFrame | None:
    """Load optional metadata table with age/sex/localization."""
    if not metadata_csv.exists():
        return None
    df = pd.read_csv(metadata_csv)
    if "image_id" not in df.columns:
        return None
    return df


def _build_split_df(
    split: str,
    metadata_dir: Path,
    aux_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """Build split dataframe from official ISIC ground-truth files."""
    split_to_csv = {
        "train": "ISIC2018_Task3_Training_GroundTruth.csv",
        "val": "ISIC2018_Task3_Validation_GroundTruth.csv",
        "test": "ISIC2018_Task3_Test_GroundTruth.csv",
    }
    gt_csv = metadata_dir / split_to_csv[split]
    gt_df = pd.read_csv(gt_csv).rename(columns={"image": "image_id"})
    for col in GT_CLASS_COLS:
        gt_df[col] = pd.to_numeric(gt_df[col], errors="coerce").fillna(0.0)
    gt_df["dx"] = gt_df[GT_CLASS_COLS].idxmax(axis=1).str.lower()
    gt_df["label"] = gt_df["dx"].map(CLASS_TO_IDX).astype(int)
    gt_df["dx_type"] = "official_split"
    gt_df["age"] = pd.NA
    gt_df["sex"] = "unknown"
    gt_df["localization"] = "unknown"

    if split == "train":
        lesion_csv = metadata_dir / "ISIC2018_Task3_Training_LesionGroupings.csv"
        lesion_df = pd.read_csv(lesion_csv).rename(columns={"image": "image_id"})
        gt_df = gt_df.merge(lesion_df[["image_id", "lesion_id"]], on="image_id", how="left")
        gt_df["lesion_id"] = gt_df["lesion_id"].fillna(gt_df["image_id"])
    else:
        gt_df["lesion_id"] = gt_df["image_id"]

    if aux_metadata is not None:
        gt_df = gt_df.merge(
            aux_metadata[["image_id", "age", "sex", "localization", "dx_type"]].drop_duplicates(
                "image_id"
            ),
            on="image_id",
            how="left",
            suffixes=("", "_aux"),
        )
        gt_df["age"] = gt_df["age_aux"].where(gt_df["age_aux"].notna(), gt_df["age"])
        gt_df["sex"] = gt_df["sex_aux"].where(gt_df["sex_aux"].notna(), gt_df["sex"])
        gt_df["localization"] = gt_df["localization_aux"].where(
            gt_df["localization_aux"].notna(), gt_df["localization"]
        )
        gt_df["dx_type"] = gt_df["dx_type_aux"].where(gt_df["dx_type_aux"].notna(), gt_df["dx_type"])
        gt_df = gt_df.drop(
            columns=["age_aux", "sex_aux", "localization_aux", "dx_type_aux"], errors="ignore"
        )

    return gt_df


def _build_loss_functions(
    cfg: DictConfig,
    class_weights: torch.Tensor | None = None,
) -> tuple[nn.Module, nn.Module | None]:
    """Create stage-1 and optional stage-2 losses (guide: mel FN 10x, BCC 5x).

    When class_weights is provided and config requests it, Focal loss uses
    inverse-frequency weights to address class imbalance.
    """
    gamma = float(OmegaConf.select(cfg, "training.training.loss.gamma", default=2.0))
    use_alpha = str(
        OmegaConf.select(cfg, "training.training.loss.alpha", default="none")
    ).lower() == "class_frequency_inverse"
    alpha = class_weights if (use_alpha and class_weights is not None) else None
    stage1 = FocalLoss(alpha=alpha, gamma=gamma)

    mel_weight = float(
        OmegaConf.select(cfg, "training.training.loss.melanoma_fn_weight", default=10.0)
    )
    bcc_weight = float(
        OmegaConf.select(cfg, "training.training.loss.bcc_fn_weight", default=5.0)
    )
    stage2 = CostSensitiveLoss(
        cost_matrix={
            "mel": {"fn_weight": mel_weight},
            "bcc": {"fn_weight": bcc_weight},
            "default": {"fn_weight": 1.0},
        }
    )
    return stage1, stage2


def _build_dataloaders(cfg: DictConfig, device: torch.device | None = None) -> tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders from ISIC 2018 splits; optionally pool train+val."""
    image_size = int(_cfg_value(cfg, "data.data.image_size", 380))
    batch_size = int(_cfg_value(cfg, "training.training.batch_size", 32))
    # CPU backward is very slow; cap batch size so backward can complete
    if device is not None and device.type == "cpu" and batch_size > 8:
        import logging
        logging.getLogger("train").info(
            "Capping batch_size to 8 for CPU (backward pass is slow with larger batches)."
        )
        batch_size = 8
    num_workers = int(_cfg_value(cfg, "data.data.num_workers", 4))
    pin_memory = bool(_cfg_value(cfg, "data.data.pin_memory", True))
    use_all_data = bool(_cfg_value(cfg, "data.data.use_all_data", False))
    use_preprocessed = bool(_cfg_value(cfg, "data.data.use_preprocessed", False))
    val_fraction = float(_cfg_value(cfg, "data.data.val_fraction", 0.15))
    validate_images_on_init = bool(_cfg_value(cfg, "data.data.validate_images_on_init", True))
    seed = int(_cfg_value(cfg, "seed", 42))

    if use_preprocessed:
        image_root = _resolve_path(
            str(_cfg_value(cfg, "data.data.preprocessed_image_dir", "data/preprocessed/images"))
        )
        apply_color_constancy = False
    else:
        image_root = _resolve_path(str(cfg.data.data.image_dir))
        apply_color_constancy = True
    metadata_csv = _resolve_path(str(cfg.data.data.metadata_csv))
    metadata_dir = metadata_csv.parent

    if not metadata_csv.exists():
        metadata_csv = metadata_dir / "metadata_merged.csv"
    aux_metadata = _load_aux_metadata(metadata_csv)

    train_df = _build_split_df(split="train", metadata_dir=metadata_dir, aux_metadata=aux_metadata)
    val_df = _build_split_df(split="val", metadata_dir=metadata_dir, aux_metadata=aux_metadata)

    if use_all_data:
        train_df["image_subdir"] = "train"
        val_df["image_subdir"] = "val"
        pool_df = pd.concat([train_df, val_df], ignore_index=True)
        groups = pool_df["lesion_id"].astype(str).to_numpy()
        indices = np.arange(len(pool_df))
        gss = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=seed)
        train_idx, val_idx = next(gss.split(indices, groups=groups))
        train_pool = pool_df.iloc[train_idx].reset_index(drop=True)
        val_pool = pool_df.iloc[val_idx].reset_index(drop=True)
        train_pool, encoding_stats = encode_metadata(train_pool)
        val_pool, _ = encode_metadata(val_pool, stats=encoding_stats)
        train_dataset = HAM10000Dataset(
            df=train_pool,
            image_dir=image_root,
            transform=get_train_transforms(
                image_size=image_size,
                apply_color_constancy=apply_color_constancy,
            ),
            validate_images_on_init=validate_images_on_init,
        )
        val_dataset = HAM10000Dataset(
            df=val_pool,
            image_dir=image_root,
            transform=get_val_transforms(
                image_size=image_size,
                apply_color_constancy=apply_color_constancy,
            ),
            validate_images_on_init=validate_images_on_init,
        )
    else:
        train_df, encoding_stats = encode_metadata(train_df)
        val_df, _ = encode_metadata(val_df, stats=encoding_stats)
        train_dataset = HAM10000Dataset(
            df=train_df,
            image_dir=image_root / "train",
            transform=get_train_transforms(
                image_size=image_size,
                apply_color_constancy=apply_color_constancy,
            ),
            validate_images_on_init=validate_images_on_init,
        )
        val_dataset = HAM10000Dataset(
            df=val_df,
            image_dir=image_root / "val",
            transform=get_val_transforms(
                image_size=image_size,
                apply_color_constancy=apply_color_constancy,
            ),
            validate_images_on_init=validate_images_on_init,
        )

    sampler = ClassBalancedSampler(
        df=train_dataset.df,
        label_col="label",
        lesion_col="lesion_id",
        batch_size=batch_size,
        seed=int(_cfg_value(cfg, "seed", 42)),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


@hydra.main(config_path="../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Train model with two-stage strategy and optional sanity check."""
    validate_config(cfg)
    seed_everything(int(cfg.seed))
    device_override = OmegaConf.select(cfg, "training.training.device", default=None)
    device = get_device(device_override if device_override is not None else None)
    logger = setup_logger("train", log_file=PROJECT_ROOT / "outputs/logs/train.log")
    logger.info("Using device: %s (override=%s)", device, device_override)
    wandb_run = setup_wandb(
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        project_name=str(OmegaConf.select(cfg, "experiment.experiment.wandb_project", default="dermafusion")),
        run_name=str(cfg.experiment_name),
    )

    train_loader, val_loader = _build_dataloaders(cfg, device=device)
    # Class imbalance: inverse-frequency weights for Focal loss (when enabled in config)
    train_dataset = train_loader.dataset
    class_weights = compute_class_weights(train_dataset.df["label"].to_numpy())
    logger.info(
        "Class weights (inverse freq): %s",
        [f"{class_weights[i].item():.3f}" for i in range(7)],
    )

    model = build_model(cfg)
    stage1_loss, stage2_loss = _build_loss_functions(cfg, class_weights=class_weights)
    optimizer = build_optimizer(model, cfg)
    scheduler_name = str(
        OmegaConf.select(cfg, "training.training.scheduler.name", default="cosine_annealing_warmup")
    )
    total_epochs = int(OmegaConf.select(cfg, "training.training.epochs", default=100))
    if scheduler_name == "cosine_warm_restarts":
        from src.training.schedulers import CosineAnnealingWarmRestarts
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=int(OmegaConf.select(cfg, "training.training.scheduler.T_0", default=10)),
            T_mult=int(OmegaConf.select(cfg, "training.training.scheduler.T_mult", default=2)),
            eta_min=float(OmegaConf.select(cfg, "training.training.scheduler.min_lr", default=1e-6)),
        )
    else:
        scheduler = CosineAnnealingWithWarmup(
            optimizer=optimizer,
            warmup_epochs=int(OmegaConf.select(cfg, "training.training.scheduler.warmup_epochs", default=5)),
            total_epochs=total_epochs,
            min_lr=float(OmegaConf.select(cfg, "training.training.scheduler.min_lr", default=1e-6)),
        )
    early_stopping = EarlyStopping(
        patience=int(OmegaConf.select(cfg, "training.training.early_stopping_patience", default=15)),
        mode="max",
    )
    checkpoint = ModelCheckpoint(save_dir=PROJECT_ROOT / "outputs/checkpoints")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        device=device,
        loss_fn=stage1_loss,
        stage2_loss_fn=stage2_loss,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping=early_stopping,
        checkpoint=checkpoint,
        wandb_cb=WandBCallback(run=wandb_run),
    )

    if bool(OmegaConf.select(cfg, "training.training.sanity_check", default=False)):
        start_loss, end_loss = trainer.run_sanity_check(iterations=50)
        logger.info("Sanity check: start_loss=%.6f end_loss=%.6f", start_loss, end_loss)
        return

    logger.info("Starting training loop (fit).")
    results: dict[str, Any] = trainer.fit()
    logger.info("Training finished with results: %s", results)


if __name__ == "__main__":
    main()
