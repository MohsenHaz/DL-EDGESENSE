"""Early fusion — concatenate before feature extraction (Equation 13, E_available < Eth1)."""
import torch, torch.nn as nn

class EarlyFusion(nn.Module):
    def __init__(self, embed_dim, n_mod=5):
        super().__init__()
        self.proj = nn.Linear(embed_dim * n_mod, embed_dim)
    def forward(self, feat_stack, energy_budget=None):
        return self.proj(feat_stack.flatten(1))
