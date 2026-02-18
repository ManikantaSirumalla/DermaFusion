"""Step 3 model architecture tests."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pytest
import torch
from omegaconf import OmegaConf

from src.models.classifier import ClassificationHead
from src.models.fusion import CrossAttentionFusion, FiLMFusion, LateFusion
from src.models.image_encoder import ImageEncoder
from src.models.metadata_encoder import MetadataEncoder
from src.models.model_factory import build_model
from src.utils.config import compose_config


@pytest.mark.parametrize("backbone", ["efficientnet_b4", "swin_tiny", "convnext_v2"])
def test_image_encoder_forward_shape(backbone: str) -> None:
    """Each configured backbone should produce a 2D feature tensor."""
    model = ImageEncoder(backbone_name=backbone, pretrained=False)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    assert y.ndim == 2
    assert y.shape[0] == 2


def test_metadata_encoder_forward_shape() -> None:
    """Metadata encoder returns expected batch x output_dim shape."""
    encoder = MetadataEncoder(num_sex_categories=3, num_loc_categories=16, output_dim=128)
    x = torch.tensor(
        [
            [0.35, 0.0, 4.0, 0.0, 0.0, 0.0],
            [0.72, 1.0, 7.0, 0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    y = encoder(x)
    assert y.shape == (2, 128)


def test_fusion_modules_shapes() -> None:
    """All fusion modules return expected output dimensions."""
    img = torch.randn(4, 256)
    meta = torch.randn(4, 128)

    late = LateFusion(256, 128, hidden_dim=512)
    film = FiLMFusion(256, 128)
    xattn = CrossAttentionFusion(256, 128, num_heads=4)

    assert late(img, meta).shape == (4, 512)
    assert film(img, meta).shape == (4, 256)
    assert xattn(img, meta).shape == (4, 256)


def test_classifier_outputs_logits() -> None:
    """Classification head output should match class dimension."""
    head = ClassificationHead(input_dim=256, num_classes=7)
    x = torch.randn(3, 256)
    logits = head(x)
    assert logits.shape == (3, 7)


def test_build_model_image_only_and_multimodal() -> None:
    """Factory should build both image-only and multimodal variants."""
    cfg_image_only = compose_config(
        config_dir=Path("configs"),
        overrides=["model.model.use_metadata=false"],
    )
    model = build_model(cfg_image_only)
    batch = {"image": torch.randn(2, 3, 224, 224)}
    assert model(batch).shape == (2, 7)

    cfg_multimodal = compose_config(
        config_dir=Path("configs"),
        overrides=["model.model.use_metadata=true", "model.model.fusion=late"],
    )
    model_mm = build_model(cfg_multimodal)
    batch_mm = {
        "image": torch.randn(2, 3, 224, 224),
        "metadata": torch.tensor(
            [[0.2, 0.0, 1.0, 0.0, 0.0, 0.0], [0.8, 1.0, 2.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        ),
    }
    assert model_mm(batch_mm).shape == (2, 7)


def test_encoder_freeze_unfreeze_behavior() -> None:
    """Encoder unfreeze should toggle gradient flags from frozen state."""
    encoder = ImageEncoder(backbone_name="efficientnet_b4", pretrained=False, freeze=True)
    assert all(not p.requires_grad for p in encoder.backbone.parameters())
    encoder.unfreeze(num_layers=3)
    assert any(p.requires_grad for p in encoder.backbone.parameters())


def test_model_state_dict_save_load_roundtrip() -> None:
    """Model should save and load with matching parameter keys."""
    cfg = OmegaConf.create(
        {
            "model": {
                "backbone": "efficientnet_b4",
                "pretrained": False,
                "use_metadata": False,
                "num_classes": 7,
            }
        }
    )
    model = build_model(cfg)
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_path = Path(tmp_dir) / "model.pt"
        torch.save(model.state_dict(), save_path)
        reloaded = build_model(cfg)
        reloaded.load_state_dict(torch.load(save_path, map_location="cpu"))
        assert set(model.state_dict().keys()) == set(reloaded.state_dict().keys())
