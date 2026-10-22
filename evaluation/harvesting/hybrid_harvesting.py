"""Hybrid harvesting — solar + RF + vibration (Table 15)."""
from .solar_harvesting import simulate_solar
from .rf_harvesting import simulate_rf
from .vibration_harvesting import simulate_vibration
import numpy as np

def total_harvested(hours=24):
    return simulate_solar(hours) + simulate_rf(hours) + simulate_vibration(hours)
