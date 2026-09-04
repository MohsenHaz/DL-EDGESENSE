"""Differential privacy — Section 4.4.8 (epsilon=1.0, delta=1e-5)."""
import torch

def clip_and_noise(model, max_grad_norm: float = 1.0,
                   noise_multiplier: float = 1.1, device="cpu"):
    """Gradient clipping + Gaussian noise for (eps, delta)-DP."""
    for param in model.parameters():
        if param.grad is not None:
            param.grad.data = torch.nn.utils.clip_grad_norm_(
                [param], max_grad_norm).to(device) * param.grad.data / (param.grad.data.norm() + 1e-8)
            param.grad.data += noise_multiplier * max_grad_norm * torch.randn_like(param.grad.data)
