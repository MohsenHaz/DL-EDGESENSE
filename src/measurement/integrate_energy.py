"""Energy integration from raw power traces (the method described in measurements/integration_method.md)."""
import numpy as np


def load_trace(path):
    """Return dict with t_s, v_V, i_A, gpio arrays and header metadata."""
    meta, skip = {}, 0
    with open(path) as f:
        for line in f:
            if line.startswith("#"):
                k, _, v = line[1:].strip().partition("=")
                meta[k] = v
                skip += 1
            else:
                break
    arr = np.genfromtxt(path, delimiter=",", names=True, skip_header=skip)
    return {"t_s": arr["t_s"], "v_V": arr["v_bus_V"], "i_A": arr["i_A"],
            "gpio": arr["gpio"].astype(int), "meta": meta}


def power_w(tr):
    return tr["v_V"] * tr["i_A"]


def trapezoid_j(t, p):
    return float(np.sum((p[1:] + p[:-1]) * 0.5 * np.diff(t)))


def effective_rate_hz(tr):
    t = tr["t_s"]
    return float((len(t) - 1) / (t[-1] - t[0]))


def idle_power_w(idle_trace):
    """Mean power of an idle-baseline trace (no inference running)."""
    return float(np.mean(power_w(idle_trace)))


def find_windows(gpio):
    """Index pairs (start, end) of contiguous gpio==1 regions (one per inference)."""
    g = np.asarray(gpio).astype(int)
    d = np.diff(np.concatenate([[0], g, [0]]))
    starts, ends = np.where(d == 1)[0], np.where(d == -1)[0] - 1
    return list(zip(starts, ends))


def per_inference(tr, idle_w):
    """Idle-subtracted energy (mJ), duration (ms), mean in-window power (mW) and max instantaneous power (mW)
    for every inference window of a session trace."""
    t, p = tr["t_s"], power_w(tr)
    out = []
    for s, e in find_windows(tr["gpio"]):
        if e <= s:
            continue
        dur = t[e] - t[s]
        gross = trapezoid_j(t[s:e + 1], p[s:e + 1])
        net = gross - idle_w * dur
        out.append({"energy_mj": net * 1e3, "latency_ms": dur * 1e3,
                    "mean_power_mw": net / dur * 1e3, "peak_power_mw": float(p[s:e + 1].max() * 1e3)})
    return out


def windowed_mean_peak_mw(tr, window_s=1.0):
    """Maximum of the window-averaged power (the statistic reported as 'peak of windowed mean')."""
    t, p = tr["t_s"], power_w(tr)
    edges = np.arange(t[0], t[-1], window_s)
    best = 0.0
    for a in edges:
        m = (t >= a) & (t < a + window_s)
        if m.sum() > 1:
            best = max(best, float(p[m].mean()))
    return best * 1e3
