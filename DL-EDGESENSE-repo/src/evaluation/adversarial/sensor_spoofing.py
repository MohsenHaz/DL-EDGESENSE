"""Sensor spoofing attacks — Section 4.4.9, Table 18."""
import torch

def replay_attack(signal: torch.Tensor, shift: float = 0.1) -> torch.Tensor:
    """Frequency-shifted replay for RF signals."""
    return torch.roll(signal, int(shift * signal.size(-1)), dims=-1)

def noise_injection(tensor: torch.Tensor, snr_db: float = 10.0) -> torch.Tensor:
    """Add Gaussian noise at target SNR."""
    signal_power = tensor.pow(2).mean()
    noise_power = signal_power / (10 ** (snr_db / 10))
    return tensor + torch.randn_like(tensor) * noise_power.sqrt()
