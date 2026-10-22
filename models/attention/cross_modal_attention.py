"""Cross-modal transformer (Equation 14) — lightweight, 4 heads."""
import torch.nn as nn

class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        out, _ = self.attn(x, x, x)
        return self.norm(x + out)
