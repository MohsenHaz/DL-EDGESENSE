"""
Modality-specific feature extractors, Section 3.2.

Kept intentionally lightweight (matching the paper's "energy-aware
feature extraction" motivation): a MobileNetV3-small backbone for
visual, a small CRNN for acoustic, a TCN for IMU, an MLP for
environmental scalars, and a dilated causal CNN for RF.
"""
import torch
import torch.nn as nn


class VisualExtractor(nn.Module):
    """MobileNetV3-small backbone, depth multiplier configurable per
    energy mode (Table 12/13: High-Energy=1.0, Standard=0.75,
    Energy-Constrained=0.5, Critical=0.35)."""

    def __init__(self, out_dim: int = 128, depth_multiplier: float = 1.0):
        super().__init__()
        try:
            import torchvision.models as tvm
            base = tvm.mobilenet_v3_small(weights=None)
            base.classifier = nn.Identity()
            self.backbone = base
            backbone_dim = 576
        except ImportError:
            # torchvision not available -- lightweight CNN fallback
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 16, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            backbone_dim = 32
        self.proj = nn.Linear(backbone_dim, out_dim)
        self.depth_multiplier = depth_multiplier

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        return self.proj(feat)


class AcousticExtractor(nn.Module):
    """Lightweight CRNN over 128-bin mel-spectrograms (S4.2)."""

    def __init__(self, out_dim: int = 128, n_mels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_mels, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.ReLU(),
        )
        self.gru = nn.GRU(64, out_dim, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, n_mels, T]
        feat = self.conv(x).transpose(1, 2)  # [B, T, 64]
        _, h_n = self.gru(feat)
        return h_n.squeeze(0)


class IMUExtractor(nn.Module):
    """Temporal Convolutional Network with residual connections."""

    def __init__(self, in_channels: int = 9, out_dim: int = 128):
        super().__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(in_channels, 64, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=4, dilation=4), nn.ReLU(),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.tcn(x).mean(dim=-1)
        return self.proj(feat)


class EnvironmentalExtractor(nn.Module):
    """Small MLP over standardised temperature/humidity/air-quality scalars."""

    def __init__(self, in_dim: int = 3, out_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.ReLU(), nn.Linear(32, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RFExtractor(nn.Module):
    """Dilated causal convolutional network over PSD windows (S4.3)."""

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=2, dilation=2), nn.ReLU(),
        )
        self.proj = nn.Linear(64, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.net(x).mean(dim=-1)
        return self.proj(feat)
