"""INA219 raw-trace logger.

Writes one CSV per session with columns: t_s, v_bus_V, i_A, gpio.
`gpio` is 1 while an inference is executing (set by the caller through mark()).
There is NO simulation fallback: if the sensor is unavailable the logger raises,
so synthetic numbers can never end up in results/.

The *effective* sampling rate is measured from the timestamps and written in the
CSV header (# effective_rate_hz=...). The INA219 conversion time and the I2C
transaction time bound the achievable rate; scripts/verify_numbers.py refuses to
claim a 10 kHz acquisition unless the recorded effective rate supports it.
"""
import csv, threading, time


class HardwareNotAvailableError(RuntimeError):
    pass


class INA219Logger:
    def __init__(self, out_csv, shunt_ohms=0.1, max_expected_amps=2.0, address=0x40, busnum=1,
                 adc_bits=9, target_rate_hz=10000):
        try:
            from ina219 import INA219  # pip install pi-ina219
        except ImportError as e:
            raise HardwareNotAvailableError("pi-ina219 not installed; cannot acquire real traces") from e
        self.out_csv = out_csv
        self.target_rate_hz = target_rate_hz
        self.ina = INA219(shunt_ohms, max_expected_amps, address=address, busnum=busnum)
        adc = {9: self.ina.ADC_9BIT, 10: self.ina.ADC_10BIT, 11: self.ina.ADC_11BIT, 12: self.ina.ADC_12BIT}[adc_bits]
        self.ina.configure(self.ina.RANGE_16V, self.ina.GAIN_AUTO, adc, adc)
        self._gpio = 0
        self._run = False
        self._rows = []
        self._thread = None

    def mark(self, state):
        self._gpio = 1 if state else 0

    def _loop(self):
        t0 = time.perf_counter()
        while self._run:
            t = time.perf_counter() - t0
            self._rows.append((t, self.ina.voltage(), self.ina.current() / 1000.0, self._gpio))

    def start(self):
        self._rows, self._run = [], True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._run = False
        self._thread.join()
        n = len(self._rows)
        if n < 2:
            raise RuntimeError("no samples captured")
        eff = (n - 1) / (self._rows[-1][0] - self._rows[0][0])
        with open(self.out_csv, "w", newline="") as f:
            f.write(f"# effective_rate_hz={eff:.1f}\n# target_rate_hz={self.target_rate_hz}\n")
            w = csv.writer(f)
            w.writerow(["t_s", "v_bus_V", "i_A", "gpio"])
            w.writerows(self._rows)
        return eff
