"""Run energy profiling across all modes and platforms."""
from src.models.model_factory import create_model
from src.utils.config_loader import load_config
from src.evaluation.energy_profiler import profile_all_modes
from data.sensor_simulation import simulate_all
import torch

def main():
    for mode in ["high_energy", "standard", "energy_constrained", "critical"]:
        config = load_config(f"configs/{mode}_mode.yaml")
        model = create_model(config).eval()
        inputs = simulate_all(1)
        energy_budget = torch.ones(1, 1)
        fn = lambda: model(inputs, energy_budget)
        results = profile_all_modes(model, lambda: fn(), [mode])
        print(f"Mode={mode}: {results}")

if __name__ == "__main__":
    main()
