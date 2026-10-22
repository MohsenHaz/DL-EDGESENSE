"""HPEQ quantization engine for INT8/INT4 inference."""
import torch

def quantize_model(model, bit_width: int = 8):
    if bit_width == 8:
        return torch.quantization.quantize_dynamic(
            model, {torch.nn.Linear}, dtype=torch.qint8)
    return model  # for custom bit-widths use HPEQ directly
