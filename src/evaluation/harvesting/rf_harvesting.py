"""RF energy harvesting simulation."""
import numpy as np

def simulate_rf(hours=24, base_mw=5.0):
    noise = np.random.uniform(0.8, 1.2, hours)
    return base_mw * noise
