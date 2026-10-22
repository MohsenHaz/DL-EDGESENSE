"""TCN-based IMU extractor with residual connections (Section 3.3)."""
import torch, torch.nn as nn

class _ResBlock(nn.Module):
    def __init__(self, ch, dilation):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(ch), nn.ReLU(),
            nn.Conv1d(ch, ch, 1),
        )
    def forward(self, x): return x + self.conv(x)

class IMUExtractor(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.stem = nn.Conv1d(9, 64, 1)  # 9-axis IMU
        self.blocks = nn.Sequential(*[_ResBlock(64, 2**i) for i in range(4)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(64, embed_dim)

    def forward(self, x):
        x = self.stem(x.permute(0, 2, 1))
        x = self.blocks(x)
        return self.proj(self.pool(x).squeeze(-1))
