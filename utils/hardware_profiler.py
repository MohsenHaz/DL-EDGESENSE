"""Hardware energy profiling wrapper around the raw-trace logger.

Replaces the earlier profiler, which (a) fell back to a hard-coded 50 mW placeholder when no sensor was present and
(b) multiplied ONE instantaneous power reading by wall-clock time. Energy is now obtained exclusively by integrating
a recorded power trace (src/measurement/integrate_energy.py).
"""
import time
from src.measurement.ina219_logger import INA219Logger, HardwareNotAvailableError  # noqa: F401


class EnergyProfiler:
    def __init__(self, out_dir, warmup_s=60, reps=100, **logger_kw):
        self.out_dir, self.warmup_s, self.reps, self.kw = out_dir, warmup_s, reps, logger_kw

    def record_idle(self, name="idle.csv", duration_s=30):
        lg = INA219Logger(f"{self.out_dir}/{name}", **self.kw)
        lg.start(); time.sleep(duration_s); return lg.stop()

    def record_session(self, fn, name):
        """Run fn() `reps` times with the GPIO marker raised around every call; returns effective sampling rate."""
        time.sleep(self.warmup_s)
        lg = INA219Logger(f"{self.out_dir}/{name}", **self.kw)
        lg.start()
        for _ in range(self.reps):
            lg.mark(1); fn(); lg.mark(0); time.sleep(0.05)
        return lg.stop()
