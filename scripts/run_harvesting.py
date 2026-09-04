"""Energy harvesting integration tests — Table 15."""
from src.evaluation.harvesting.hybrid_harvesting import total_harvested
import numpy as np

def main():
    harvested = total_harvested(hours=24)
    print(f"24h harvested: {np.sum(harvested):.1f} mWh")
    print(f"Peak: {np.max(harvested):.1f} mW  Mean: {np.mean(harvested):.1f} mW")

if __name__ == "__main__":
    main()
