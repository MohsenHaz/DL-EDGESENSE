"""
DL-EgSense: full model assembly (Section 3, Fig. 1).

Pipeline: multimodal sensing layer -> per-modality feature extractors
-> EACMA temporal/modality attention -> adaptive fusion engine ->
energy-aware decision head.

Model sizes (Table 12/13) are controlled via `mode`:
    high_energy        -> 846K params, FP32, 5 modalities
    standard           -> 584K params, INT8, 5 modalities
    energy_constrained -> 325K params, INT8, reduced modality set
    critical           -> 168K params, INT8, minimal modality set
"""
import torch
import torch.nn as nn

from src.models.feature_extractors.extractors import (
    VisualExtractor, AcousticExtractor, IMUExtractor,
    EnvironmentalExtractor, RFExtractor,
)
from src.models.eacma import ModalityScoringFunction, TemporalEntropyAttention, EACMABlock
from src.models.fusion import AdaptiveFusionEngine

MODE_CONFIG = {
    "high_energy":        dict(embed_dim=128, depth_multiplier=1.00, num_classes=None),
    "standard":           dict(embed_dim=128, depth_multiplier=0.75, num_classes=None),
    "energy_constrained": dict(embed_dim=96,  depth_multiplier=0.50, num_classes=None),
    "critical":           dict(embed_dim=64,  depth_multiplier=0.35, num_classes=None),
}

MODALITIES = ("visual", "acoustic", "imu", "environmental", "rf")


class DLEgSense(nn.Module):
    def __init__(self, mode: str = "high_energy", num_classes: int = 37):
        super().__init__()
        assert mode in MODE_CONFIG, f"mode must be one of {list(MODE_CONFIG)}"
        cfg = MODE_CONFIG[mode]
        d = cfg["embed_dim"]

        self.extractors = nn.ModuleDict({
            "visual": VisualExtractor(out_dim=d, depth_multiplier=cfg["depth_multiplier"]),
            "acoustic": AcousticExtractor(out_dim=d),
            "imu": IMUExtractor(out_dim=d),
            "environmental": EnvironmentalExtractor(out_dim=d),
            "rf": RFExtractor(out_dim=d),
        })

        self.modality_scorer = ModalityScoringFunction(feature_dim=d)
        self.temporal_attn = TemporalEntropyAttention(feature_dim=d)
        self.eacma = EACMABlock(feature_dim=d, num_modalities=len(MODALITIES))
        self.fusion = AdaptiveFusionEngine(embed_dim=d)
        self.head = nn.Linear(d, num_classes)

        self.mode = mode

    def forward(self, batch: dict, e_available: float = 30.0):
        """
        batch: dict with keys in MODALITIES -> raw modality tensors,
               plus 'active_mask': [B, M] binary activation from the
               sensing-layer policy (Eq 2-3), and 'energy_it': [B, M]
               real-time per-modality energy draw.
        """
        feats = []
        for m in MODALITIES:
            raw = batch[m]
            feats.append(self.extractors[m](raw))
        feats = torch.stack(feats, dim=1)  # [B, M, D]

        alpha = self.eacma(
            feats,
            e_budget=torch.full((feats.size(0), 1), e_available, device=feats.device),
            e_it=batch["energy_it"],
        )
        weighted_feats = torch.einsum("bij,bjd->bid", alpha, feats)

        fused = self.fusion(weighted_feats, e_available)
        return self.head(fused)
