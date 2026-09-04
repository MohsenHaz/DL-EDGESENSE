"""Spatial attention for visual data (Equation 10)."""
import torch, torch.nn as nn

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, F_visual):
        """F_visual: (B, C, H, W)"""
        pool = F_visual.mean(dim=1, keepdim=True)
        return torch.sigmoid(self.conv(pool))
