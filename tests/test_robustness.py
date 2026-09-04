"""Robustness tests — adversarial scenarios."""
import torch
from src.evaluation.adversarial.sensor_spoofing import noise_injection, replay_attack
from src.evaluation.adversarial.data_poisoning import flip_labels
from src.evaluation.adversarial.entropy_detection import modality_entropy, detect_anomaly

def test_noise_injection():
    x = torch.ones(10, 512)
    out = noise_injection(x, snr_db=10)
    assert not torch.allclose(x, out)

def test_flip_labels():
    t = torch.zeros(100, dtype=torch.long)
    flipped = flip_labels(t, num_classes=5, flip_rate=0.1)
    assert (flipped != t).sum() <= 15  # roughly 10%

def test_entropy_detection():
    logits = torch.zeros(4, 12); logits[:2, 0] = 10.0  # confident
    entropy = modality_entropy(logits)
    flags = detect_anomaly(entropy, threshold=1.0)
    assert flags[2:].all()   # high-entropy → flagged
