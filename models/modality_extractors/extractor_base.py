"""Base extractor class."""
import torch.nn as nn

class BaseExtractor(nn.Module):
    def __init__(self, output_dim: int):
        super().__init__()
        self.output_dim = output_dim
