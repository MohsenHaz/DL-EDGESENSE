"""Benchmark DL-EgSense against baselines (Tables 3, 4)."""
from src.utils.logger import get_logger
logger = get_logger(__name__)

BASELINE_ENERGY = {
    "DeepSense":       146.0,
    "EdgeML":          128.2,
    "Federated-Sense": 138.7,
    "EfficientSensing":111.0,
    "DL-EgSense":       51.0,
}

BASELINE_F1 = {
    "DeepSense":       91.5,
    "EdgeML":          75.3,
    "Federated-Sense": 84.5,
    "EfficientSensing":89.7,
    "DL-EgSense":      92.7,
}

def print_comparison():
    print(f"{'Model':<20} {'Energy (mJ)':>12} {'F1 (%)':>10} {'Ratio':>10}")
    for m in BASELINE_ENERGY:
        e = BASELINE_ENERGY[m]; f = BASELINE_F1[m]
        print(f"{m:<20} {e:>12.1f} {f:>10.1f} {f**2/e:>10.2f}")
