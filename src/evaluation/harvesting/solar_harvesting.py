"""Solar harvesting simulation — Table 15."""
import numpy as np

def simulate_solar(hours: int = 24, panel_area_m2: float = 0.02,
                   efficiency: float = 0.18) -> np.ndarray:
    """Simulate hourly harvested power (mW) over 24h."""
    irradiance = np.maximum(0, np.sin(np.linspace(0, np.pi, hours)) * 800)
    return irradiance * panel_area_m2 * efficiency * 1000  # mW
