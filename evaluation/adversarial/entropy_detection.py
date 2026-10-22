"""Entropy-based anomaly detection — Section 4.4.9, Table 18."""
import torch, torch.nn.functional as F

def modality_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy of softmax output."""
    p = F.softmax(logits, dim=-1)
    return -(p * torch.log(p + 1e-8)).sum(dim=-1)

def detect_anomaly(entropy: torch.Tensor, threshold: float = 1.5) -> torch.Tensor:
    """Flag modalities with entropy above threshold."""
    return entropy > threshold
