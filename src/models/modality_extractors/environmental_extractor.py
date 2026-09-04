"""Environmental sensor (temperature, humidity, air quality) extractor."""
import torch.nn as nn

class EnvironmentalExtractor(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 32), nn.ReLU(),
            nn.Linear(32, 64), nn.ReLU(),
            nn.Linear(64, embed_dim),
        )
    def forward(self, x): return self.net(x)
