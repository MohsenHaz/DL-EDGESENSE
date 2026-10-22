"""Benchmark against the retained baselines (results/baselines/*.json + measured energy)."""
from src.evaluation.benchmark import print_comparison

if __name__ == "__main__":
    print_comparison()
