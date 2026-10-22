"""Ablation variants for Table 19. Each variant changes the model (or the inference harness) for real."""
import torch, torch.nn as nn
from .fusion.early_fusion import EarlyFusion

VARIANTS = ["full", "no_eacma", "no_hpeq", "no_adaptive_sensing", "no_energy_training",
            "no_cross_modal_fusion", "visual_only", "acoustic_only", "standard_attention"]


class _SkipAttention(nn.Module):
    def forward(self, q, k, v, energy_budget=None, modality_energies=None):
        return q, None


class _StdAttention(nn.Module):
    def __init__(self, d, h, p=0.1):
        super().__init__()
        self.m = nn.MultiheadAttention(d, h, dropout=p, batch_first=True)

    def forward(self, q, k, v, energy_budget=None, modality_energies=None):
        o, w = self.m(q, k, v, need_weights=True)
        return o, w


class _MeanFusion(nn.Module):
    def forward(self, x, energy_budget=None):
        return x.mean(dim=1)


def apply_ablation(model, name, config=None):
    """Return (model, config_overrides). `no_energy_training` needs retraining with lambda_e = 0."""
    overrides = {}
    d, h = model.embed_dim, model.eacma.num_heads
    if name == "no_eacma":
        model.eacma = _SkipAttention()
    elif name == "standard_attention":
        model.eacma = _StdAttention(d, h)
    elif name == "no_hpeq":
        model.classifier[0] = nn.Linear(d, d // 2)
    elif name == "no_cross_modal_fusion":
        model.eacma = _SkipAttention()
        model.fusion = _MeanFusion()
    elif name == "no_energy_training":
        overrides = {"training": {"lambda_e": 0.0, "lambda_c": 0.0}}
    elif name not in VARIANTS:
        raise ValueError(name)
    return model, overrides


def active_modalities(model, inputs, name, feats=None, thr=0.5):
    """Sensor selection used by the evaluation harness (the model forward itself consumes whatever it is given)."""
    if name == "visual_only":
        return {"visual": inputs["visual"]}
    if name == "acoustic_only":
        return {"acoustic": inputs["acoustic"]}
    if name == "no_adaptive_sensing" or feats is None:
        return inputs
    gate = torch.sigmoid(model.activation_policy(torch.cat(feats, dim=-1)))   # (B, 5)
    keep = (gate.mean(0) > thr).tolist()
    sel = {m: inputs[m] for m, k in zip(model.MODALITIES, keep) if k and m in inputs}
    return sel or inputs
