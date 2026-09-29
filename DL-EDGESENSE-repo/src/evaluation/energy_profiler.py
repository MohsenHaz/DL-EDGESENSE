"""Energy profiling — INA219 @ 10 kHz, 100 reps, idle-subtracted."""
from src.utils.hardware_profiler import EnergyProfiler

def profile_all_modes(model, input_fn, modes=None):
    modes = modes or ["high_energy", "standard", "energy_constrained", "critical"]
    results = {}
    for mode in modes:
        profiler = EnergyProfiler()
        idle = profiler.measure_idle()
        mean_mj, std_mj = profiler.profile_inference(lambda: model(input_fn()), idle)
        results[mode] = {"mean_mj": mean_mj, "std_mj": std_mj}
    return results
