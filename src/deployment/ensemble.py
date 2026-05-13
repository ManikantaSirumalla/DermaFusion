"""Soft-vote ensemble used by both the Gradio runtime and deployment bundles.

The forward returns ``log(weighted_avg_softmax)`` so that
``softmax(forward(x))`` recovers the ensemble probability vector exactly.
That keeps the Gradio runtime's existing softmax-after-forward path correct
without special-casing the ensemble.

The module shape (``members``, ``weights``, ``class_thresholds``, plus
non-buffer attributes ``class_names``, ``mean``, ``std``, ``img_size``)
matches the pickled instance saved into
``app/deployment_Bundle/dermafusion_ensemble.pt``.
"""

from __future__ import annotations

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class DermaFusionEnsemble(nn.Module):
    """Weighted soft-vote of N timm classifiers.

    Buffers (saved in state_dict):
        members          - nn.ModuleList of timm classifiers
        weights          - 1-D tensor of per-member soft-vote weights
        class_thresholds - 1-D tensor of per-class probability thresholds
                           consumed by downstream decision logic (not used
                           inside forward)

    Attributes (pickled, not in state_dict):
        class_names, mean, std, img_size
    """

    members: nn.ModuleList
    weights: torch.Tensor
    class_thresholds: torch.Tensor

    def __init__(
        self,
        backbones: list[str] | None = None,
        weights: list[float] | None = None,
        class_thresholds: list[float] | None = None,
        class_names: list[str] | None = None,
        mean: list[float] | None = None,
        std: list[float] | None = None,
        img_size: int = 380,
    ) -> None:
        super().__init__()
        backbones = list(backbones or [])
        weights = list(weights) if weights is not None else [1.0] * len(backbones)
        if len(weights) != len(backbones):
            raise ValueError("weights and backbones must have equal length")

        n_cls = len(class_names) if class_names else 8
        self.members = nn.ModuleList(
            [
                timm.create_model(b, pretrained=False, num_classes=n_cls)
                for b in backbones
            ]
        )

        w = torch.as_tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", w / w.sum().clamp_min(1e-8))

        thrs = torch.as_tensor(class_thresholds or [0.5] * n_cls, dtype=torch.float32)
        self.register_buffer("class_thresholds", thrs)

        self.class_names = list(class_names) if class_names else []
        self.mean = list(mean) if mean else [0.485, 0.456, 0.406]
        self.std = list(std) if std else [0.229, 0.224, 0.225]
        self.img_size = int(img_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg: torch.Tensor | None = None
        for i, m in enumerate(self.members):
            probs = F.softmax(m(x), dim=1)
            term = self.weights[i] * probs
            avg = term if avg is None else avg + term
        assert avg is not None
        return torch.log(avg.clamp_min(1e-9))


# Back-compat alias kept so older bundles tagged ``model_loader == "ensemble"``
# still resolve to the same class via the inference runtime loader.
EnsembleSoftVote = DermaFusionEnsemble


def register_for_unpickling() -> None:
    """Make ``DermaFusionEnsemble`` discoverable from ``__main__``.

    The deployment bundle was pickled from a Colab kernel where the class
    lived in ``__main__``. ``torch.load`` will look it up there during
    reconstruction; without this hook we'd get
    ``AttributeError: Can't get attribute 'DermaFusionEnsemble' on '__main__'``.
    """
    import sys

    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "DermaFusionEnsemble"):
        main_mod.DermaFusionEnsemble = DermaFusionEnsemble
