"""Tests for attention mechanisms."""
import torch, pytest
from src.models.attention.modality_attention import ModalityAttention
from src.models.attention.temporal_attention import TemporalAttention
from src.models.attention.spatial_attention import SpatialAttention

def test_modality_attention():
    ma = ModalityAttention(64)
    h = torch.randn(2, 5, 64)
    w = ma(h)
    assert w.shape == (2, 5)
    assert torch.allclose(w.sum(-1), torch.ones(2), atol=1e-5)

def test_temporal_attention():
    ta = TemporalAttention(64)
    h = torch.randn(2, 20, 64)
    w = ta(h)
    assert w.shape == (2, 20)

def test_spatial_attention():
    sa = SpatialAttention()
    F = torch.randn(2, 64, 7, 7)
    mask = sa(F)
    assert mask.shape == (2, 1, 7, 7)
    assert (mask >= 0).all() and (mask <= 1).all()
