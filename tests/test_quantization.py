"""Quantization tests."""
import torch
from src.models.hpeq import quantize

def test_quantize_1bit():
    x = torch.randn(100)
    xq = quantize(x, 1)
    assert xq.unique().numel() <= 2

def test_quantize_8bit():
    x = torch.randn(1000)
    xq = quantize(x, 8)
    assert xq.unique().numel() <= 256
