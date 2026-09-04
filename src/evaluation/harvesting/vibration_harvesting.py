"""Vibration/acoustic harvesting simulation."""
import numpy as np

def simulate_vibration(hours=24, base_mw=3.0):
    return base_mw * np.abs(np.sin(np.linspace(0, 4*np.pi, hours)))
