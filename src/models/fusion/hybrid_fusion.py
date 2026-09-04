"""Hybrid fusion — default strategy (Equation 14, cross-modal transformer)."""
import torch, torch.nn as nn
from ..attention.cross_modal_attention import CrossModalAttention

class HybridFusion(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.cross_attn = CrossModalAttention(embed_dim, num_heads)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, feat_stack, energy_budget=None):
        """feat_stack: (B, N, D)"""
        fused = self.cross_attn(feat_stack)                 # (B, N, D)
        return self.pool(fused.permute(0, 2, 1)).squeeze(-1) # (B, D)
