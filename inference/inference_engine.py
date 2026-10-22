"""Inference engine — batch size 1 for edge deployment."""
import torch
from src.utils.timer import Timer

class InferenceEngine:
    def __init__(self, model, device="cpu"):
        self.model = model.eval().to(device)
        self.device = device

    def infer(self, inputs: dict, energy_budget=None):
        with torch.no_grad(), Timer() as t:
            logits, attn = self.model(inputs, energy_budget)
        pred = logits.argmax(1)
        return {"prediction": pred, "confidence": logits.softmax(1).max(1).values,
                "latency_ms": t.elapsed_ms, "attention": attn}
