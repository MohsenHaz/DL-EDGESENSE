# Energy integration method
1. Idle power `P_idle` = mean(v*i) over the idle trace of the same platform and session.
2. Inference windows = contiguous runs of `gpio == 1` (the GPIO line is raised immediately before and lowered immediately after each forward pass).
3. Gross energy of a window = trapezoidal integral of `p(t) = v_bus * i` over the window (`numpy` trapezoid on the recorded timestamps).
4. Per-inference energy = gross energy - `P_idle * window duration` (idle-subtracted). Units: J -> mJ.
5. Mean in-inference power = per-inference energy / window duration. Instantaneous peak = max of `p(t)` inside the window.
6. "Peak of windowed mean" (24 h runs) = max over 1-s windows of the mean power; it is *not* an instantaneous peak and is labelled as such in the tables.
7. Per-configuration statistics: mean, median, std over all windows of all sessions; the effective sampling rate of every trace is reported (`effective_rate_hz_min`).
Implemented in `src/measurement/integrate_energy.py` and tested in `tests/test_energy_integration.py`.
