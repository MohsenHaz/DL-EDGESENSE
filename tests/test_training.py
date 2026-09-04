"""Training pipeline smoke test."""
import torch
from src.models.dl_egsense import DLEgSense
from src.training.loss_functions import EnergyAwareLoss

def test_backward_pass():
    model = DLEgSense(num_classes=12, embed_dim=32)
    crit  = EnergyAwareLoss()
    opt   = torch.optim.Adam(model.parameters())

    inputs = {"environmental": torch.randn(2, 3)}
    targets = torch.randint(0, 12, (2,))
    alpha = torch.rand(2)
    energies = torch.rand(2, 1)

    logits, _ = model(inputs)
    loss = crit(logits, targets, alpha, energies, model)
    opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() > 0
