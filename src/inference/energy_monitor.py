"""Real-time energy monitoring during inference."""
from src.utils.hardware_profiler import EnergyProfiler

class EnergyMonitor:
    def __init__(self):
        self.profiler = EnergyProfiler()
        self._idle = 0.0

    def calibrate(self):
        self._idle = self.profiler.measure_idle()

    def measure(self, fn):
        mean_mj, std_mj = self.profiler.profile_inference(fn, self._idle)
        return mean_mj, std_mj
