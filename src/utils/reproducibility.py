"""Fix all random seeds for reproducibility (seeds: 0, 42, 123, 456, 789)."""
import random, os
import numpy as np
import torch

SEEDS = [0, 42, 123, 456, 789]

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
