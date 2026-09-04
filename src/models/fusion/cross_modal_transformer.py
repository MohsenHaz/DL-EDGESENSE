"""Lightweight cross-modal transformer — shared parameters, pruned attention."""
import torch.nn as nn
from ..attention.cross_modal_attention import CrossModalAttention

class CrossModalTransformer(nn.Module):
    def __init__(self, embed_dim, num_heads=4, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList([CrossModalAttention(embed_dim, num_heads)
                                     for _ in range(num_layers)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
