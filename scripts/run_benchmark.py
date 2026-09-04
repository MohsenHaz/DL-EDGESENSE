"""Benchmark against all baselines."""
from src.evaluation.benchmark import print_comparison

def main():
    print("\n=== DL-EgSense vs Baselines ===")
    print_comparison()
    print("\nBaseline implementations in src/evaluation/baselines/")

if __name__ == "__main__":
    main()
