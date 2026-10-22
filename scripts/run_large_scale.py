"""Large-scale (1000-device) scenario. Energy per device is taken from measured per-inference energy; no placeholders."""
import numpy as np
from src.evaluation.results_io import load_energy_summary, energy_mj

def main(platform="rpi4b", mode="standard", n_devices=1000, seed=0):
    e = energy_mj(load_energy_summary(), platform, mode)          # raises MissingMeasurementError if not measured
    sd = energy_mj(load_energy_summary(), platform, mode, "energy_mj_std")
    rng = np.random.default_rng(seed)
    per_dev = rng.normal(e, sd, n_devices).clip(min=0)
    print(f"{n_devices} devices, mean {per_dev.mean():.2f} mJ, std {per_dev.std():.2f} mJ (sampled from measured distribution)")

if __name__ == "__main__":
    main()
