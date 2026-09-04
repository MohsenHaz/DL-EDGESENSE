"""Data augmentation techniques."""
import torch, random

def random_modality_dropout(sample: dict, p: float = 0.2) -> dict:
    """Randomly deactivate modalities during training."""
    out = dict(sample)
    out["inputs"] = {k: v for k, v in sample["inputs"].items()
                     if random.random() > p}
    return out

def add_gaussian_noise(tensor: torch.Tensor, sigma: float = 0.01) -> torch.Tensor:
    return tensor + torch.randn_like(tensor) * sigma

def random_temporal_crop(imu: torch.Tensor, target_len: int = 80) -> torch.Tensor:
    if imu.size(0) <= target_len:
        return imu
    start = random.randint(0, imu.size(0) - target_len)
    return imu[start:start + target_len]
