"""
DL-EgSense — main model class.
Integrates EACMA + HPEQ + Hierarchical Attention + Adaptive Fusion.
"""
import torch, torch.nn as nn
from .eacma import EACMA
from .hpeq import HPEQLayer
from .modality_extractors.visual_extractor import VisualExtractor
from .modality_extractors.acoustic_extractor import AcousticExtractor
from .modality_extractors.imu_extractor import IMUExtractor
from .modality_extractors.rf_extractor import RFExtractor
from .modality_extractors.environmental_extractor import EnvironmentalExtractor
from .fusion.hybrid_fusion import HybridFusion


class DLEgSense(nn.Module):
    """DL-EgSense: Energy-Aware Multimodal Sensing Framework."""

    MODALITIES = ["visual", "acoustic", "imu", "environmental", "rf"]

    def __init__(self, num_classes: int, embed_dim: int = 128,
                 num_heads: int = 4, dropout: float = 0.1,
                 lambda_e: float = 1.0, lambda_c: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.lambda_e = lambda_e
        self.lambda_c = lambda_c

        # Modality-specific feature extractors
        self.extractors = nn.ModuleDict({
            "visual":        VisualExtractor(embed_dim),
            "acoustic":      AcousticExtractor(embed_dim),
            "imu":           IMUExtractor(embed_dim),
            "environmental": EnvironmentalExtractor(embed_dim),
            "rf":            RFExtractor(embed_dim),
        })

        # Sensor activation policy (Equations 2-3)
        self.activation_policy = nn.Linear(embed_dim * 5, 5)

        # EACMA — energy-adaptive cross-modal attention (Equation 11)
        self.eacma = EACMA(embed_dim, num_heads, dropout)

        # Adaptive fusion (Section 3.7, Equation 13)
        self.fusion = HybridFusion(embed_dim, num_heads)

        # Classifier with HPEQ
        self.classifier = nn.Sequential(
            HPEQLayer(embed_dim, embed_dim // 2, context_dim=embed_dim,
                      lambda_energy=lambda_e),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, num_classes),
        )

    def forward(self, inputs: dict,
                energy_budget: torch.Tensor = None,
                modality_energies: torch.Tensor = None):
        """
        Args:
            inputs: dict of {modality_name: tensor}
            energy_budget: (B, 1) normalised energy [0, 1]
            modality_energies: (B, N_modalities) per-modality power
        Returns:
            logits: (B, num_classes)
            attn_weights: attention weights for interpretability
        """
        # Extract features from each active modality
        feats = []
        for mod in self.MODALITIES:
            if mod in inputs:
                feats.append(self.extractors[mod](inputs[mod]))  # (B, D)

        # Stack → (B, N, D)
        feat_stack = torch.stack(feats, dim=1)

        # EACMA cross-modal attention (Equation 11)
        fused, attn = self.eacma(feat_stack, feat_stack, feat_stack,
                                  energy_budget=energy_budget,
                                  modality_energies=modality_energies)

        # Adaptive fusion (Equation 13)
        pooled = self.fusion(fused, energy_budget)

        # Classification
        logits = self.classifier(pooled)
        return logits, attn

    def count_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
