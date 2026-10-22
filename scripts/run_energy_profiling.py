"""Record raw INA219 traces for all modes on the connected board. Needs the physical sensor (no fallback)."""
import argparse, os, torch
from src.models.model_factory import create_model
from src.utils.config_loader import load_config
from src.evaluation.energy_profiler import record_all_modes
from data.sensor_simulation import simulate_all   # synthetic INPUT tensors only; energy is measured, not simulated

MODES = ["high_energy", "standard", "energy_constrained", "critical", "recovery"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", required=True, choices=["rpi4b", "jetson_nano"])
    ap.add_argument("--session", type=int, default=1)
    a = ap.parse_args()
    out = f"measurements/{a.platform}"; os.makedirs(out, exist_ok=True)
    for mode in MODES:
        cfg = load_config(f"configs/{mode}_mode.yaml")
        model = create_model(cfg).eval()
        with torch.no_grad():
            rates = record_all_modes(model, lambda: simulate_all(1), out, [mode], a.session)
        print(mode, "effective sampling rate [Hz]:", rates)
    print("Now append the new files to measurements/MANIFEST.csv and run `make aggregate`.")

if __name__ == "__main__":
    main()
