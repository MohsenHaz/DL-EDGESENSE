"""
Hybrid Precision-Energy Quantisation (HPEQ) -- Section 3.3 / 3.7.

Eq (4): Q(x,b) = 1/(2^b-1) * round((2^b-1) * (x - x_min)/(x_max - x_min))
                 * (x_max - x_min) + x_min
Eq (5)-(6): per-layer bit-width search under a computational energy budget.
Eq (12): joint precision-energy objective per layer/modality.
"""
import torch
import torch.nn as nn

ALLOWED_BITS = (1, 2, 4, 8, 16)


class STEQuantize(torch.autograd.Function):
    """Straight-through estimator: forward quantises, backward passes
    the gradient through unchanged (standard practice for QAT)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, bits: int):
        x_min = x.min()
        x_max = x.max()
        span = (x_max - x_min).clamp_min(1e-8)
        levels = (2 ** bits) - 1
        x_norm = (x - x_min) / span
        x_q = torch.round(x_norm * levels) / levels
        return x_q * span + x_min

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def quantize(x: torch.Tensor, bits: int) -> torch.Tensor:
    """Eq (4), exactly as specified: a function of both x and b."""
    assert bits in ALLOWED_BITS, f"bits must be one of {ALLOWED_BITS}"
    return STEQuantize.apply(x, bits)


class HPEQLayer(nn.Module):
    """
    Wraps a linear/conv layer with per-layer, context-dependent
    bit-width selection (Eq 12). `context_predictor` maps a scenario
    embedding to a soft distribution over ALLOWED_BITS; at inference
    the argmax bit-width is used, at train time a soft mixture keeps
    the search differentiable.
    """

    def __init__(self, layer: nn.Module, context_dim: int = 8):
        super().__init__()
        self.layer = layer
        self.context_predictor = nn.Sequential(
            nn.Linear(context_dim, 16),
            nn.ReLU(),
            nn.Linear(16, len(ALLOWED_BITS)),
        )

    def select_bitwidth(self, context: torch.Tensor) -> int:
        logits = self.context_predictor(context)
        idx = torch.argmax(logits, dim=-1).mode().values.item()
        return ALLOWED_BITS[idx]

    def forward(self, x: torch.Tensor, context: torch.Tensor):
        bits = self.select_bitwidth(context)
        q_weight = quantize(self.layer.weight, bits)
        return nn.functional.linear(x, q_weight, self.layer.bias), bits


def per_op_energy(bits: int) -> float:
    """Empirical energy-per-op lookup (mJ / 1k ops), calibrated on the
    RPi 4B / Jetson Nano per Section 3.7. Placeholder values -- replace
    with device-specific calibration constants before regenerating
    Table 3/4/8/12/13 numbers on new hardware."""
    return {1: 0.0006, 2: 0.0009, 4: 0.0015, 8: 0.0026, 16: 0.0048}[bits]
