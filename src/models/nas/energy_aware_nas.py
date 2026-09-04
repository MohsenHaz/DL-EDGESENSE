"""Energy-Aware NAS — Section 3.9, Equations 19-22.

Objective: alpha* = argmax_alpha E_{a~A(alpha)}[R(a) - lambda * E(a)]
Gradient:  nabla_alpha L_val(w*(alpha), alpha) - lambda * nabla_alpha E(alpha)
"""
import torch, torch.nn as nn, torch.nn.functional as F

class EnergyAwareNAS(nn.Module):
    def __init__(self, num_ops: int = 4, lambda_energy: float = 0.3):
        super().__init__()
        self.lambda_energy = lambda_energy
        # Architecture parameters alpha (Equation 22 — differentiable relaxation)
        self.arch_params = nn.ParameterDict({
            "embedding": nn.Parameter(torch.zeros(4)),    # [32,64,128,256]
            "heads":     nn.Parameter(torch.zeros(4)),    # [1,2,4,8]
            "bits":      nn.Parameter(torch.zeros(4)),    # [2,4,8,16]
            "fusion":    nn.Parameter(torch.zeros(3)),    # [early,hybrid,late]
        })
        self.embed_choices  = [32, 64, 128, 256]
        self.head_choices   = [1, 2, 4, 8]
        self.bit_choices    = [2, 4, 8, 16]
        self.energy_mapping = {32: 0.3, 64: 0.5, 128: 1.0, 256: 2.0}

    def mixed_op(self, x, alpha):
        """Differentiable mixture — Equation 22."""
        return F.softmax(alpha, dim=0)

    def energy_model(self, alpha_embed):
        """E(alpha) = E_sensing + E_computation + E_memory (Equation 21)."""
        probs = F.softmax(alpha_embed, dim=0)
        costs = torch.tensor([self.energy_mapping[c] for c in self.embed_choices],
                             device=alpha_embed.device)
        return (probs * costs).sum()

    def arch_loss(self):
        """Energy regularisation term for NAS gradient (Equation 20)."""
        return self.lambda_energy * self.energy_model(self.arch_params["embedding"])

    def get_best_arch(self):
        return {
            "embed_dim": self.embed_choices[self.arch_params["embedding"].argmax().item()],
            "num_heads": self.head_choices[self.arch_params["heads"].argmax().item()],
            "bits":      self.bit_choices[self.arch_params["bits"].argmax().item()],
            "fusion":    ["early","hybrid","late"][self.arch_params["fusion"].argmax().item()],
        }
