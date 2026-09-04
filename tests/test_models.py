"""DL-EgSense model architecture tests."""
import torch, pytest
from src.models.dl_egsense import DLEgSense

@pytest.fixture
def model(): return DLEgSense(num_classes=12, embed_dim=64, num_heads=2)

def test_forward_pass(model):
    inputs = {
        "visual": torch.randn(2, 3, 224, 224),
        "acoustic": torch.randn(2, 1, 64, 128),
        "imu": torch.randn(2, 100, 9),
        "environmental": torch.randn(2, 3),
        "rf": torch.randn(2, 512),
    }
    logits, attn = model(inputs)
    assert logits.shape == (2, 12)

def test_energy_budget_routing(model):
    inputs = {"environmental": torch.randn(2, 3)}
    e_budget = torch.tensor([[0.2], [0.8]])
    logits, _ = model(inputs, energy_budget=e_budget)
    assert logits.shape == (2, 12)

def test_param_count(model):
    assert model.count_params() > 0
