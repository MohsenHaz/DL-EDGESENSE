"""Unit tests for HPEQ (Equation 12)."""
import torch, pytest
from src.models.hpeq import HPEQLayer, quantize

@pytest.fixture
def hpeq(): return HPEQLayer(64, 32, context_dim=16)

def test_quantize_passthrough():
    x = torch.randn(10)
    assert torch.allclose(quantize(x, 16), x)

def test_quantize_reduces_precision():
    x = torch.randn(100) * 10
    xq = quantize(x, 2)
    assert xq.unique().numel() <= 4  # at most 2^2=4 unique values

def test_hpeq_output_shape(hpeq):
    x = torch.randn(4, 64)
    ctx = torch.randn(4, 16)
    out = hpeq(x, ctx)
    assert out.shape == (4, 32)

def test_hpeq_loss(hpeq):
    ctx = torch.randn(4, 16)
    loss = hpeq.hpeq_loss(ctx)
    assert loss.item() > 0
