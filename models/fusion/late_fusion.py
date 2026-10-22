"""Late fusion — fuse high-level representations (Equation 13, E_available >= Eth2)."""
import torch, torch.nn as nn

class LateFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Linear(embed_dim, 1)

    def forward(self, feat_stack, energy_budget=None):
        gates = torch.softmax(self.gate(feat_stack).squeeze(-1), dim=-1)
        return (feat_stack * gates.unsqueeze(-1)).sum(dim=1)
