"""Data pipeline tests."""
import torch
from data.sensor_simulation import simulate_all
from data.preprocessing import preprocess_imu, preprocess_rf

def test_simulate_all():
    inputs = simulate_all(B=2)
    assert "visual" in inputs
    assert inputs["visual"].shape == (2, 3, 224, 224)
    assert inputs["acoustic"].shape == (2, 1, 64, 128)
    assert inputs["imu"].shape == (2, 100, 9)

def test_preprocess_imu():
    data = torch.randn(100, 9) * 10 + 5
    out = preprocess_imu(data)
    assert abs(out.mean().item()) < 0.1  # near zero mean

def test_preprocess_rf():
    sig = torch.randn(512) * 5 + 2
    out = preprocess_rf(sig)
    assert abs(out.mean().item()) < 0.1
