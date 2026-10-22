"""NAS search space definition."""
SEARCH_SPACE = {
    "embedding_dims":    [32, 64, 128, 256],
    "num_heads":         [1, 2, 4, 8],
    "bit_widths":        [2, 4, 8, 16],
    "fusion_strategies": ["early", "hybrid", "late"],
    "num_layers":        [1, 2, 3, 4],
}
