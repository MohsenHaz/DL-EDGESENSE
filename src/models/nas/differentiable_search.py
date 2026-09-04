"""DARTS-inspired differentiable search with energy term (Equation 20)."""
import torch

def nas_step(val_loss_fn, arch_module, w_optimizer, arch_optimizer):
    """Single NAS bilevel optimisation step."""
    arch_optimizer.zero_grad()
    val_loss = val_loss_fn() + arch_module.arch_loss()
    val_loss.backward()
    arch_optimizer.step()
    return val_loss.item()
