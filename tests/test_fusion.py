"""Tests for fusion strategies."""
import torch
from src.models.fusion.early_fusion import EarlyFusion
from src.models.fusion.hybrid_fusion import HybridFusion
from src.models.fusion.late_fusion import LateFusion

def test_early_fusion():
    B, N, D = 2, 5, 64
    f = EarlyFusion(D, n_mod=N)
    out = f(torch.randn(B, N, D))
    assert out.shape == (B, D)

def test_hybrid_fusion():
    B, N, D = 2, 5, 64
    f = HybridFusion(D)
    out = f(torch.randn(B, N, D))
    assert out.shape == (B, D)

def test_late_fusion():
    B, N, D = 2, 5, 64
    f = LateFusion(D)
    out = f(torch.randn(B, N, D))
    assert out.shape == (B, D)
