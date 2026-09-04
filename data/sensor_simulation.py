"""Simulate sensor data for testing and ablation."""
import torch

def simulate_visual(B=1):
    return torch.randn(B, 3, 224, 224)

def simulate_acoustic(B=1):
    return torch.randn(B, 1, 64, 128)

def simulate_imu(B=1, T=100):
    return torch.randn(B, T, 9)

def simulate_environmental(B=1):
    return torch.rand(B, 3) * torch.tensor([40.0, 100.0, 500.0])  # T(°C), RH(%), AQI

def simulate_rf(B=1, N=512):
    return torch.randn(B, N)

def simulate_all(B=1):
    return {
        "visual": simulate_visual(B),
        "acoustic": simulate_acoustic(B),
        "imu": simulate_imu(B),
        "environmental": simulate_environmental(B),
        "rf": simulate_rf(B),
    }
