"""Anomaly detection evaluation — Table 16."""
import torch
from src.models.model_factory import create_model
from src.utils.config_loader import load_config
from src.evaluation.adversarial.entropy_detection import modality_entropy, detect_anomaly
from data.sensor_simulation import simulate_all

def main():
    config = load_config("configs/standard_mode.yaml")
    model = create_model(config).eval()
    inputs = simulate_all(8)
    with torch.no_grad():
        logits, _ = model(inputs)
    entropy = modality_entropy(logits)
    flags   = detect_anomaly(entropy)
    print(f"Anomaly recall simulation: {flags.float().mean().item()*100:.1f}%")

if __name__ == "__main__":
    main()
