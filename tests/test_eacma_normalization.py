"""
Regression test for the Equation (9) normalisation bug flagged across
several review rounds: the entropy penalty must be applied BEFORE the
softmax so that sum_j beta_{i,j} == 1 exactly.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from src.models.eacma import TemporalEntropyAttention


def test_beta_sums_to_one():
    torch.manual_seed(0)
    attn = TemporalEntropyAttention(feature_dim=16, gamma=0.3)
    h = torch.randn(4, 10, 16)  # [B, T, D]
    beta = attn(h)
    row_sums = beta.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), \
        f"Attention weights do not sum to 1: {row_sums}"


def test_quantizer_is_function_of_x_and_b():
    from src.models.hpeq import quantize
    x = torch.randn(100)
    q4 = quantize(x, 4)
    q8 = quantize(x, 8)
    assert not torch.allclose(q4, q8), "Q(x,b) must depend on b"
    assert not torch.allclose(q4, torch.zeros_like(q4)), "Q(x,b) must depend on x"


if __name__ == "__main__":
    test_beta_sums_to_one()
    test_quantizer_is_function_of_x_and_b()
    print("All tests passed.")
