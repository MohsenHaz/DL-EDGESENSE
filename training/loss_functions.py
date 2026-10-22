"""
Energy-aware loss functions — Section 3.8, Equations 15-18.

L_total = L_task + lambda_e * L_energy + lambda_c * L_complexity  (Eq 15)
L_energy = sum_i(alpha_i * e_i) + sum_l C_l(W_l)                  (Eq 16)
L_complexity = sum_l ||W_l||_1                                      (Eq 17)
e_i_soft = e_i * sigma(k * (alpha_i - tau))                        (Eq 18)
"""
import torch, torch.nn as nn

class EnergyAwareLoss(nn.Module):
    def __init__(self, lambda_e: float = 1.0, lambda_c: float = 0.1,
                 k: float = 10.0, tau: float = 0.5):
        super().__init__()
        self.lambda_e = lambda_e
        self.lambda_c = lambda_c
        self.k = k
        self.tau = tau
        self.task_loss = nn.CrossEntropyLoss()

    def energy_loss(self, alpha: torch.Tensor, energies: torch.Tensor,
                    model_weights) -> torch.Tensor:
        """Equation 16 — sensing + computation energy."""
        # Soft energy activation (Equation 18)
        e_soft = energies * torch.sigmoid(self.k * (alpha - self.tau))
        sensing = (alpha * e_soft).sum(dim=-1).mean()
        compute = sum(w.abs().mean() for w in model_weights if w.requires_grad)
        return sensing + compute * 0.01

    def complexity_loss(self, model_weights) -> torch.Tensor:
        """Equation 17 — L1 sparsity."""
        return sum(w.abs().sum() for w in model_weights if w.requires_grad) * 1e-6

    def forward(self, logits, targets, alpha, energies, model):
        weights = list(model.parameters())
        l_task = self.task_loss(logits, targets)
        l_energy = self.energy_loss(alpha, energies, weights)
        l_complex = self.complexity_loss(weights)
        return l_task + self.lambda_e * l_energy + self.lambda_c * l_complex
