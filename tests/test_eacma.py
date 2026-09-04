"""Unit tests for EACMA (Equation 11)."""
import torch, pytest
from src.models.eacma import EACMA

@pytest.fixture
def eacma(): return EACMA(embed_dim=64, num_heads=4)

def test_output_shape(eacma):
    B, N, D = 2, 5, 64
    q = k = v = torch.randn(B, N, D)
    out, attn = eacma(q, k, v)
    assert out.shape == (B, N, D)
    assert attn.shape == (B, 4, N, N)

def test_energy_budget_effect(eacma):
    B, N, D = 2, 5, 64
    x = torch.randn(B, N, D)
    e_high = torch.ones(B, 1)
    e_low  = torch.ones(B, 1) * 0.1
    out_h, _ = eacma(x, x, x, energy_budget=e_high)
    out_l, _ = eacma(x, x, x, energy_budget=e_low)
    assert not torch.allclose(out_h, out_l), "Energy budget should change attention"

def test_modality_energy_effect(eacma):
    B, N, D = 2, 5, 64
    x = torch.randn(B, N, D)
    e_mod = torch.rand(B, N)
    out, _ = eacma(x, x, x, modality_energies=e_mod)
    assert out.shape == (B, N, D)
