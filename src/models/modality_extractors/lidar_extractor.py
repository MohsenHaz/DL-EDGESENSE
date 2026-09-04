"""LiDAR feature extractor — 7-modality config (Velodyne VLP-16)."""
import torch.nn as nn

class LiDARExtractor(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(3, 64, 1), nn.ReLU(),
            nn.Conv1d(64, 128, 1), nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),
            nn.Flatten(), nn.Linear(128, embed_dim),
        )
    def forward(self, x): return self.net(x.permute(0, 2, 1))
