"""MobileNetV3-based visual feature extractor (Section 3.3)."""
import torch, torch.nn as nn
import torchvision.models as tv

class VisualExtractor(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        backbone = tv.mobilenet_v3_small(weights=None)
        # Replace classifier
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.proj = nn.Linear(576, embed_dim)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        return self.proj(x)
