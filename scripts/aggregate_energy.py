"""traces (measurements/MANIFEST.csv) -> results/energy_summary.csv"""
from src.measurement.aggregate import aggregate

if __name__ == "__main__":
    print("wrote", aggregate())
