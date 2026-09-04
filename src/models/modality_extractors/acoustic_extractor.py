"""CRNN-based acoustic feature extractor — mel-spectrogram input (Section 3.3)."""
import torch.nn as nn

class AcousticExtractor(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.rnn = nn.GRU(64 * 4, embed_dim // 2, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B = x.size(0)
        x = self.cnn(x)                             # (B, C, H, W)
        x = x.flatten(2).permute(0, 2, 1)           # (B, T, C*H)
        _, h = self.rnn(x)
        h = h.permute(1, 0, 2).contiguous().view(B, -1)
        return self.proj(h)
