"""Modality-level attention (Equations 7-8)."""
import torch, torch.nn as nn, torch.nn.functional as F

class ModalityAttention(nn.Module):
    """alpha_i = softmax(f_theta(h_i, e_i))  — Equation 7."""
    def __init__(self, embed_dim):
        super().__init__()
        self.Wh = nn.Linear(embed_dim, embed_dim, bias=False)
        self.We = nn.Linear(1, embed_dim, bias=False)
        self.W  = nn.Linear(embed_dim, 1, bias=True)

    def forward(self, h, e=None):
        score = torch.tanh(self.Wh(h))                          # (B,N,D)
        if e is not None:
            score = score + self.We(e.unsqueeze(-1))
        return F.softmax(self.W(score).squeeze(-1), dim=-1)     # (B,N)
