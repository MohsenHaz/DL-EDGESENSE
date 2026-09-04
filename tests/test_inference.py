"""Inference engine tests."""
import torch
from src.models.dl_egsense import DLEgSense
from src.inference.inference_engine import InferenceEngine
from data.sensor_simulation import simulate_all

def test_inference_returns_prediction():
    model = DLEgSense(num_classes=12, embed_dim=32)
    engine = InferenceEngine(model)
    inputs = simulate_all(1)
    result = engine.infer(inputs)
    assert "prediction" in result
    assert "latency_ms" in result
    assert result["prediction"].shape == (1,)
