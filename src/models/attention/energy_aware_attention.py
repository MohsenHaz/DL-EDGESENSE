"""Energy-aware scoring function f_theta(h_i, e_i) — Equation 8."""
import torch, torch.nn as nn

class EnergyAwareScoring(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.Wh = nn.Linear(embed_dim, embed_dim)
        self.We = nn.Linear(1, embed_dim)
        self.WT = nn.Linear(embed_dim, 1)

    def forward(self, h, e):
        return self.WT(torch.tanh(self.Wh(h) + self.We(e.unsqueeze(-1))))
