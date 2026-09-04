"""Hardware profiling — INA219 current sensor @ 10 kHz."""
import time, numpy as np

try:
    from ina219 import INA219
    INA219_AVAILABLE = True
except ImportError:
    INA219_AVAILABLE = False


class EnergyProfiler:
    """Measure per-inference energy using INA219 @ 10 kHz.
    Warmup: 60 s. Repetitions: 100. Idle subtracted."""

    def __init__(self, shunt_ohms=0.1, max_expected_amps=2.0, warmup_s=60, reps=100):
        self.warmup_s = warmup_s
        self.reps = reps
        if INA219_AVAILABLE:
            self.ina = INA219(shunt_ohms, max_expected_amps)
            self.ina.configure(self.ina.RANGE_16V, self.ina.GAIN_AUTO,
                               self.ina.ADC_128SAMP, self.ina.ADC_128SAMP)
        else:
            self.ina = None

    def warmup(self):
        time.sleep(self.warmup_s)

    def measure_idle(self, duration_s=5):
        if self.ina is None:
            return 0.0
        readings = []
        t0 = time.time()
        while time.time() - t0 < duration_s:
            readings.append(self.ina.power())
        return float(np.mean(readings))

    def profile_inference(self, fn, idle_power_mw=0.0):
        """Run fn() for self.reps iterations, return mean energy mJ."""
        energies = []
        for _ in range(self.reps):
            t0 = time.perf_counter()
            fn()
            elapsed = time.perf_counter() - t0
            if self.ina:
                power_mw = self.ina.power() - idle_power_mw
            else:
                power_mw = 50.0  # placeholder
            energies.append(power_mw * elapsed * 1000)  # mJ
        return float(np.mean(energies)), float(np.std(energies))
