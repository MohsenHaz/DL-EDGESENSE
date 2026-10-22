"""Aggregate raw traces listed in measurements/MANIFEST.csv into results/energy_summary.csv.

MANIFEST columns: file,platform,config,role,session,table
  role in {idle, inference, run24h};  config = mode name, baseline name or ablation variant.
"""
import csv, os
import numpy as np
from . import integrate_energy as ie


class ManifestError(RuntimeError):
    pass


def read_manifest(root="measurements"):
    p = os.path.join(root, "MANIFEST.csv")
    if not os.path.exists(p):
        raise ManifestError(f"{p} not found: raw INA219 traces have not been added to this package yet")
    with open(p) as f:
        rows = [r for r in csv.DictReader(f) if r and not r["file"].startswith("#")]
    if not rows:
        raise ManifestError("MANIFEST.csv has no rows")
    return rows


def aggregate(root="measurements", out_csv="results/energy_summary.csv"):
    rows = read_manifest(root)
    idle = {}
    for r in rows:
        if r["role"] == "idle":
            idle[r["platform"]] = ie.idle_power_w(ie.load_trace(os.path.join(root, r["file"])))
    groups = {}
    for r in rows:
        if r["role"] != "inference":
            continue
        if r["platform"] not in idle:
            raise ManifestError(f"no idle trace for platform {r['platform']}")
        tr = ie.load_trace(os.path.join(root, r["file"]))
        for w in ie.per_inference(tr, idle[r["platform"]]):
            groups.setdefault((r["platform"], r["config"]), []).append((r["session"], w, ie.effective_rate_hz(tr)))
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["platform", "config", "n_sessions", "n_inferences", "energy_mj_mean", "energy_mj_median",
                    "energy_mj_std", "latency_ms_mean", "mean_in_inference_power_mw", "max_instantaneous_power_mw",
                    "effective_rate_hz_min"])
        for (plat, cfg), items in sorted(groups.items()):
            e = np.array([x[1]["energy_mj"] for x in items])
            w.writerow([plat, cfg, len({x[0] for x in items}), len(e), f"{e.mean():.4f}", f"{np.median(e):.4f}",
                        f"{e.std(ddof=1) if len(e) > 1 else 0:.4f}",
                        f"{np.mean([x[1]['latency_ms'] for x in items]):.3f}",
                        f"{np.mean([x[1]['mean_power_mw'] for x in items]):.2f}",
                        f"{max(x[1]['peak_power_mw'] for x in items):.2f}",
                        f"{min(x[2] for x in items):.1f}"])
    return out_csv
