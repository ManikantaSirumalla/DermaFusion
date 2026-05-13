#!/usr/bin/env python3
# DermaFusion CoreML export -- run ONCE on macOS.
#
# Place this script in a folder with E0.pt and E3.pt (downloaded from
# Drive /deployment_bundle/v_final/checkpoints/) then:
#
#   pip install 'coremltools>=7.2' 'timm' 'torch' 'numpy'
#   python export_coreml.py
#
# Produces: E0.mlpackage, E3.mlpackage, ensemble_config.json

import json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import timm, coremltools as ct
from pathlib import Path

MEMBERS = [{"name": "E0", "backbone": "efficientnet_b4", "ckpt_filename": "E0.pt"}, {"name": "E3", "backbone": "tf_efficientnet_b4.ns_jft_in1k", "ckpt_filename": "E3.pt"}]
CLASS_NAMES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC", "OTHER"]
MALIGNANT_IDXS = [0, 2, 3]
WEIGHTS = [0.7048312609138294, 0.29516873908617064]
CLASS_THRS = {"MEL": 0.29, "NV": 0.21999999999999997, "BCC": 0.18, "AKIEC": 0.25999999999999995, "BKL": 0.26999999999999996, "DF": 0.57, "VASC": 0.57, "OTHER": 0.25999999999999995}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 380

class CoreMLWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, x):
        return F.softmax(self.model(x), dim=1)

_ENSEMBLE_BUNDLE_PATH = Path('dermafusion_ensemble.pt')
_ENSEMBLE_CACHE = {'sd': None}


def _ensure_ensemble_bundle():
    """Load member sub-states from dermafusion_ensemble.pt once.

    Falls back to per-member E*.pt files only when the ensemble bundle is
    missing. The bundle path is preferred because the per-member checkpoints
    in this folder were pickled with numpy 2.x and don't load cleanly on a
    numpy 1.x environment without unsafe module aliases that crash torch.
    """
    if _ENSEMBLE_CACHE['sd'] is not None:
        return _ENSEMBLE_CACHE['sd']
    if _ENSEMBLE_BUNDLE_PATH.exists():
        import sys
        import torch.nn as _nn

        # Stand-in for the pickled class so torch.load can reconstruct without
        # importing the project's runtime; we only read state_dict afterwards.
        class DermaFusionEnsemble(_nn.Module):
            def __init__(self, *a, **k):
                super().__init__()

            def forward(self, x):
                return x

        sys.modules['__main__'].DermaFusionEnsemble = DermaFusionEnsemble
        bundle = torch.load(_ENSEMBLE_BUNDLE_PATH, map_location='cpu', weights_only=False)
        full_sd = bundle['model'].state_dict()
        _ENSEMBLE_CACHE['sd'] = full_sd
        return full_sd
    return None


def _strip_member_prefix(full_sd, member_idx):
    prefix = f'members.{member_idx}.'
    return {
        k[len(prefix):]: v
        for k, v in full_sd.items()
        if k.startswith(prefix)
    }


def load_member(member):
    model = timm.create_model(member['backbone'], pretrained=False, num_classes=len(CLASS_NAMES))

    bundle_sd = _ensure_ensemble_bundle()
    if bundle_sd is not None:
        idx = MEMBERS.index(member)
        sd = _strip_member_prefix(bundle_sd, idx)
        if sd:
            model.load_state_dict(sd, strict=False)
            return model.eval()

    ck = torch.load(member['ckpt_filename'], map_location='cpu', weights_only=False)
    sd = ck.get('model_state_dict') or ck.get('state_dict') or ck
    sd = {(k[7:] if k.startswith('module.') else k): v for k,v in sd.items()}
    model.load_state_dict(sd, strict=False)
    return model.eval()

def export(member):
    print(f'[export]   {member["name"]} loading + tracing ...')
    pt = load_member(member)
    wrapper = CoreMLWrapper(pt).eval()
    example = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    traced = torch.jit.trace(wrapper, example, strict=False)
    scale_per_ch = 1.0 / (255.0 * np.array(IMAGENET_STD))
    bias = (-np.array(IMAGENET_MEAN) / np.array(IMAGENET_STD)).tolist()
    image_scale = float(scale_per_ch.mean())
    print(f'[export]   converting {member["name"]} (FP16) ...')
    mlmodel = ct.convert(
        traced,
        inputs=[ct.ImageType(
            name="image", shape=(1, 3, IMG_SIZE, IMG_SIZE),
            scale=image_scale, bias=bias, color_layout=ct.colorlayout.RGB)],
        outputs=[ct.TensorType(name="probabilities")],
        compute_precision=ct.precision.FLOAT16,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.iOS16,
    )
    mlmodel.short_description = f'DermaFusion member {member["name"]}'
    mlmodel.author = 'DermaFusion v_final'
    mlmodel.license = 'MIT'
    out = Path(member['name'] + '.mlpackage')
    if out.exists():
        import shutil; shutil.rmtree(out)
    mlmodel.save(str(out))
    print(f'[export]   saved {out}')

for m in MEMBERS:
    export(m)

config = {
    'version': 'v_final',
    'class_names': CLASS_NAMES,
    'malignant_idxs': MALIGNANT_IDXS,
    'img_size': IMG_SIZE,
    'preprocessing': {
        'mean': IMAGENET_MEAN, 'std': IMAGENET_STD,
        'apply_shades_of_gray': True, 'apply_dullrazor': True,
        'note': 'shades-of-gray + DullRazor are NOT baked into the CoreML graph; apply them in Swift before the model call.',
    },
    'members': [{'name': m['name'], 'mlpackage': m['name']+'.mlpackage', 'backbone': m['backbone']} for m in MEMBERS],
    'ensemble_weights': WEIGHTS,
    'class_thresholds_max_bal_acc': CLASS_THRS,
}
Path('ensemble_config.json').write_text(json.dumps(config, indent=2))
print('[export] DONE. Files: ' + ', '.join(m['name']+'.mlpackage' for m in MEMBERS) + ', ensemble_config.json')
