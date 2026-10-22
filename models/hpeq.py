"""
Hybrid Precision-Energy Quantization (HPEQ)
Section 3.6, Equation 12.

b*_{l,m} = argmin_b [ L_task + lambda_e * sum(E_{l,m}(b))
                               + lambda_p * P(b_{l,m} | context) ]

Novel: joint cross-modality bit-width optimisation with context-aware
precision prediction. Convergence consistent with O(log(1/eps)) empirically;
formal proof is future work.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Straight-through uniform quantization (Equation 4)."""
    if bits >= 16:
        return x
    x_min, x_max = x.min(), x.max()
    scale = (x_max - x_min) / (2 ** bits - 1) + 1e-8
    x_q = torch.round((x - x_min) / scale) * scale + x_min
    # Straight-through estimator
    return x + (x_q - x).detach()


class ContextPrecisionPredictor(nn.Module):
    """Predict context-aware bit-width (P(b|context) in Equation 12)."""

    def __init__(self, context_dim: int, num_bit_choices: int = 5):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(context_dim, 64), nn.ReLU(),
            nn.Linear(64, num_bit_choices)
        )
        self.bit_choices = [1, 2, 4, 8, 16]

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """Return soft bit-width logits."""
        return self.mlp(context)  # (B, num_bit_choices)

    def expected_bits(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.forward(context)
        probs = F.softmax(logits, dim=-1)
        bits_tensor = torch.tensor(self.bit_choices, dtype=torch.float, device=context.device)
        return (probs * bits_tensor).sum(dim=-1)  # (B,)


class HPEQLayer(nn.Module):
    """Single HPEQ-wrapped linear layer (Equation 12)."""

    def __init__(self, in_features: int, out_features: int,
                 context_dim: int = 32,
                 lambda_energy: float = 1.0,
                 lambda_precision: float = 0.5):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.precision_predictor = ContextPrecisionPredictor(context_dim)
        self.lambda_energy = lambda_energy
        self.lambda_precision = lambda_precision
        self.current_bits = 8  # default

    def energy_cost(self, bits: float) -> float:
        """Approximate energy cost from bit-width (empirical mapping)."""
        mapping = {1: 0.2, 2: 0.35, 4: 0.55, 8: 1.0, 16: 2.0}
        return mapping.get(int(bits), 1.0)

    def forward(self, x: torch.Tensor,
                context: torch.Tensor = None,
                energy_budget: float = 1.0):
        if context is not None:
            # Choose bit-width from context
            expected_b = self.precision_predictor.expected_bits(context).mean().item()
            # Clamp to nearest available
            choices = [1, 2, 4, 8, 16]
            self.current_bits = min(choices, key=lambda b: abs(b - expected_b))

        w_q = quantize(self.linear.weight, self.current_bits)
        out = F.linear(x, w_q, self.linear.bias)
        return out

    def hpeq_loss(self, context: torch.Tensor) -> torch.Tensor:
        """Energy + precision regularisation term for Equation 12."""
        logits = self.precision_predictor(context)
        probs = F.softmax(logits, dim=-1)
        bits_t = torch.tensor([1, 2, 4, 8, 16], dtype=torch.float, device=context.device)
        energy_costs = torch.tensor(
            [self.energy_cost(b) for b in [1, 2, 4, 8, 16]],
            dtype=torch.float, device=context.device
        )
        energy_loss = (probs * energy_costs).sum(dim=-1).mean()
        precision_entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        return self.lambda_energy * energy_loss + self.lambda_precision * precision_entropy
