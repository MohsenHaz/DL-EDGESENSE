"""Base model class."""
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    def count_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
