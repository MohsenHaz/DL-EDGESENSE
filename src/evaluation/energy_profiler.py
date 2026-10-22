"""Record raw INA219 sessions for each operating mode (energy is computed later by scripts/aggregate_energy.py)."""
from src.utils.hardware_profiler import EnergyProfiler


def record_all_modes(model, input_fn, out_dir, modes, session_id=1):
    prof = EnergyProfiler(out_dir)
    prof.record_idle()
    rates = {}
    for mode in modes:
        rates[mode] = prof.record_session(lambda: model(input_fn()), f"{mode}_session_{session_id:03d}.csv")
    return rates
