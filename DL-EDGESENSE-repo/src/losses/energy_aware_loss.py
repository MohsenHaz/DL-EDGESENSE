"""
Energy-aware training objective -- Section 3.8, Eq (15)-(18).

L_total = L_task + lambda_e * L_energy + lambda_c * L_complexity   (15)
L_energy = sum_i alpha_i * e_i + sum_l C_l[W_l]                    (16)
L_complexity = sum_l ||W_l||_1                                     (17)
e~_i = e_i * sigmoid(k * (alpha_i - tau))   (soft/differentiable)  (18)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_energy(e_i: torch.Tensor, alpha_i: torch.Tensor, k: float = 10.0,
                 tau: float = 0.5) -> torch.Tensor:
    """Eq (18): differentiable relaxation of the energy indicator."""
    return e_i * torch.sigmoid(k * (alpha_i - tau))


def energy_loss(alpha: torch.Tensor, e_i: torch.Tensor,
                 per_layer_costs: torch.Tensor) -> torch.Tensor:
    """Eq (16). alpha, e_i: [B, M]; per_layer_costs: [L] computational
    cost C_l[W_l] (precomputed from per-layer MAC profiling)."""
    sensing_term = (alpha * e_i).sum(dim=-1).mean()
    compute_term = per_layer_costs.sum()
    return sensing_term + compute_term


def complexity_loss(model: nn.Module) -> torch.Tensor:
    """Eq (17): L1 norm over all trainable weights."""
    l1 = sum(w.abs().sum() for name, w in model.named_parameters()
              if w.requires_grad and "bias" not in name)
    return l1


class EnergyAwareTotalLoss(nn.Module):
    def __init__(self, task_type: str = "classification",
                 lambda_energy: float = 1.0, lambda_complexity: float = 0.1):
        super().__init__()
        assert task_type in ("classification", "regression")
        self.task_type = task_type
        self.lambda_e = lambda_energy
        self.lambda_c = lambda_complexity

    def forward(self, logits, targets, model, alpha, e_i, per_layer_costs):
        if self.task_type == "classification":
            l_task = F.cross_entropy(logits, targets)
        else:
            l_task = F.mse_loss(logits, targets)

        l_energy = energy_loss(alpha, e_i, per_layer_costs)
        l_complexity = complexity_loss(model)

        total = l_task + self.lambda_e * l_energy + self.lambda_c * l_complexity
        return total, {
            "task_loss": l_task.item(),
            "energy_loss": l_energy.item(),
            "complexity_loss": l_complexity.item(),
        }
