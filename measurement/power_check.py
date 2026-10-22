"""Consistency checks between energy, latency and reported power statistics."""


def implied_power_mw(energy_mj, latency_ms):
    return energy_mj / latency_ms * 1000.0


def check_instantaneous_peak(energy_mj, latency_ms, reported_peak_mw):
    """A true *instantaneous* peak cannot be below the mean power implied by energy/latency."""
    need = implied_power_mw(energy_mj, latency_ms)
    return reported_peak_mw >= need, need


def check_label(stat_label):
    """Windowed-mean statistics must not be labelled 'peak' without the 'windowed' qualifier."""
    s = stat_label.lower()
    return not (("peak" in s) and ("windowed" not in s) and ("instantaneous" not in s))
