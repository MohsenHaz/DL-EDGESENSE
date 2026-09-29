"""
Adaptive Fusion Engine -- Section 3.6, Eq (13)-(14).

Hybrid fusion (default) uses a lightweight cross-modal transformer:
4 attention heads, embedding dim 128, 2 encoder layers, feed-forward
dim 256, dropout 0.1, with attention-matrix pruning that retains only
the top-50% of cross-modal interaction scores (Table S2).
"""
import torch
import torch.nn as nn


class PrunedTransformerEncoderLayer(nn.Module):
    """A standard transformer encoder layer, but with attention scores
    pruned to the top-50% of interactions before the softmax-weighted
    sum, matching the energy-driven sparsification described in the
    manuscript (Section 3.6 / 3.7)."""

    def __init__(self, embed_dim=128, num_heads=4, ff_dim=256, dropout=0.1,
                 prune_ratio=0.5):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads,
                                           dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim), nn.ReLU(),
            nn.Linear(ff_dim, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.prune_ratio = prune_ratio

    def _prune(self, attn_weights: torch.Tensor) -> torch.Tensor:
        # attn_weights: [B, T, T] (averaged over heads by nn.MultiheadAttention)
        k = max(1, int(attn_weights.shape[-1] * self.prune_ratio))
        topk_vals, topk_idx = torch.topk(attn_weights, k, dim=-1)
        pruned = torch.zeros_like(attn_weights)
        pruned.scatter_(-1, topk_idx, topk_vals)
        pruned = pruned / pruned.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return pruned

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True,
                                            average_attn_weights=True)
        attn_weights = self._prune(attn_weights)
        attn_out = torch.bmm(attn_weights, x)
        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class CrossModalTransformer(nn.Module):
    def __init__(self, embed_dim=128, num_heads=4, num_layers=2, ff_dim=256,
                 dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            PrunedTransformerEncoderLayer(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, modality_feats: torch.Tensor) -> torch.Tensor:
        # modality_feats: [B, num_modalities, embed_dim]
        x = modality_feats
        for layer in self.layers:
            x = layer(x)
        return x.mean(dim=1)  # fused representation


class AdaptiveFusionEngine(nn.Module):
    """Selects early / hybrid / late fusion per Eq (13) based on the
    available energy budget relative to two thresholds."""

    def __init__(self, embed_dim=128, e_threshold_1=15.0, e_threshold_2=35.0):
        super().__init__()
        self.hybrid = CrossModalTransformer(embed_dim=embed_dim)
        self.e_threshold_1 = e_threshold_1
        self.e_threshold_2 = e_threshold_2

    def forward(self, modality_feats: torch.Tensor, e_available: float):
        if e_available < self.e_threshold_1:
            return modality_feats.mean(dim=1)          # early fusion
        elif e_available < self.e_threshold_2:
            return self.hybrid(modality_feats)          # hybrid (default)
        else:
            return modality_feats.max(dim=1).values     # late fusion
