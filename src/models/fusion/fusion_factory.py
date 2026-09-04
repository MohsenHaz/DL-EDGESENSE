"""Fusion strategy selector — Equation 13."""
from .early_fusion import EarlyFusion
from .hybrid_fusion import HybridFusion
from .late_fusion import LateFusion

def get_fusion(strategy: str, embed_dim: int, num_heads: int = 4):
    if strategy == "early":   return EarlyFusion(embed_dim)
    if strategy == "hybrid":  return HybridFusion(embed_dim, num_heads)
    if strategy == "late":    return LateFusion(embed_dim)
    raise ValueError(f"Unknown fusion strategy: {strategy}")
