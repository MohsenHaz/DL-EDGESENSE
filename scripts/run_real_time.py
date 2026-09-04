"""Real-time performance tests — Table 7."""
import torch, time
from src.models.model_factory import create_model
from src.utils.config_loader import load_config
from data.sensor_simulation import simulate_all

def main():
    config = load_config("configs/standard_mode.yaml")
    model = create_model(config).eval()
    for _ in range(100):
        inputs = simulate_all(1)
        t0 = time.perf_counter()
        with torch.no_grad(): model(inputs)
        print(f"Latency: {(time.perf_counter()-t0)*1000:.1f} ms")

if __name__ == "__main__":
    main()
