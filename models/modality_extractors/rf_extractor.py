"""Dilated causal convolution RF extractor (Section 3.3)."""
import torch.nn as nn

class RFExtractor(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=4, dilation=4), nn.ReLU(),
            nn.Conv1d(64, embed_dim, 3, padding=8, dilation=8), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
    def forward(self, x):
        return self.net(x.unsqueeze(1)).squeeze(-1)
