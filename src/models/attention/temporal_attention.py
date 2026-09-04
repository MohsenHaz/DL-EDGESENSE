"""Temporal attention with entropy penalty (Equation 9)."""
import torch, torch.nn as nn, torch.nn.functional as F

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, gamma=0.1):
        super().__init__()
        self.gamma = gamma
        self.score = nn.Linear(embed_dim, 1)

    def forward(self, h):
        """h: (B, T, D)"""
        raw = self.score(h).squeeze(-1)                         # (B, T)
        beta = F.softmax(raw, dim=-1)
        entropy = -(beta * torch.log(beta + 1e-8)).sum(-1, keepdim=True)
        beta = beta - self.gamma * entropy
        return F.softmax(beta, dim=-1)
