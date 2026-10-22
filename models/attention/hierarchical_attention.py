"""Hierarchical attention — modality + temporal + spatial (Section 3.4)."""
import torch, torch.nn as nn, torch.nn.functional as F
from .modality_attention import ModalityAttention
from .temporal_attention import TemporalAttention
from .spatial_attention import SpatialAttention

class HierarchicalAttention(nn.Module):
    def __init__(self, embed_dim, gamma=0.1):
        super().__init__()
        self.mod_attn  = ModalityAttention(embed_dim)
        self.temp_attn = TemporalAttention(embed_dim, gamma)
        self.spat_attn = SpatialAttention()

    def forward(self, feat_stack, energies=None):
        """feat_stack: (B, N_mod, D)"""
        mod_weights = self.mod_attn(feat_stack, energies)
        return feat_stack * mod_weights.unsqueeze(-1)
