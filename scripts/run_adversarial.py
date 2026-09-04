"""Run adversarial robustness tests — Table 18."""
import torch
from src.models.model_factory import create_model
from src.utils.config_loader import load_config
from src.evaluation.adversarial.sensor_spoofing import noise_injection
from src.evaluation.adversarial.data_poisoning import flip_labels
from src.evaluation.adversarial.entropy_detection import modality_entropy, detect_anomaly
from src.evaluation.adversarial.recovery_module import recover
from data.sensor_simulation import simulate_all

def main():
    config = load_config("configs/standard_mode.yaml")
    model = create_model(config).eval()

    B = 4
    inputs = simulate_all(B)
    energy_budget = torch.ones(B, 1)

    # Sensor spoofing
    inputs_spoofed = dict(inputs)
    inputs_spoofed["rf"] = noise_injection(inputs["rf"], snr_db=5)

    with torch.no_grad():
        logits, attn = model(inputs_spoofed, energy_budget)

    entropies = modality_entropy(logits)
    anomalies = detect_anomaly(entropies)
    print(f"Anomalies detected: {anomalies.sum().item()}/{B}")

    if anomalies.any():
        logits_clean, _ = recover(model, inputs, ["rf"], energy_budget)
        print(f"Recovery prediction: {logits_clean.argmax(1).tolist()}")

if __name__ == "__main__":
    main()
