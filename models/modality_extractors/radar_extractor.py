"""mmWave radar extractor — 7-modality config (TI IWR6843)."""
import torch.nn as nn

class RadarExtractor(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Linear(32, embed_dim),
        )
    def forward(self, x): return self.net(x.unsqueeze(1))
