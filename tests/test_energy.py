"""Tests for energy model and loss functions."""
import torch
from src.training.loss_functions import EnergyAwareLoss
from src.models.dl_egsense import DLEgSense

def test_energy_loss_positive():
    model = DLEgSense(num_classes=12, embed_dim=32)
    criterion = EnergyAwareLoss(lambda_e=1.0, lambda_c=0.1)
    logits = torch.randn(4, 12)
    targets = torch.randint(0, 12, (4,))
    alpha = torch.rand(4)
    energies = torch.rand(4, 5) * 0.1
    loss = criterion(logits, targets, alpha, energies, model)
    assert loss.item() > 0

def test_loss_decreases_with_lower_energy():
    model = DLEgSense(num_classes=12, embed_dim=32)
    criterion = EnergyAwareLoss(lambda_e=1.0, lambda_c=0.0)
    logits = torch.randn(4, 12)
    targets = torch.randint(0, 12, (4,))
    alpha = torch.ones(4) * 0.5
    e_high = torch.ones(4, 5) * 1.0
    e_low  = torch.ones(4, 5) * 0.1
    loss_h = criterion(logits, targets, alpha, e_high, model)
    loss_l = criterion(logits, targets, alpha, e_low, model)
    assert loss_l.item() < loss_h.item()
