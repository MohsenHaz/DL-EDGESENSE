"""Data poisoning — 10% label flipping (Section 4.4.9)."""
import torch, random

def flip_labels(targets: torch.Tensor, num_classes: int, flip_rate: float = 0.1):
    """Randomly flip flip_rate fraction of labels."""
    out = targets.clone()
    n_flip = int(len(targets) * flip_rate)
    idx = random.sample(range(len(targets)), n_flip)
    for i in idx:
        out[i] = random.randint(0, num_classes - 1)
    return out
