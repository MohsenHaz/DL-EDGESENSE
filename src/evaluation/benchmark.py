"""Baseline comparison built only from measured/aggregated results (no constants)."""
from .results_io import load_energy_summary, load_json_dir, energy_mj


def comparison(platform="rpi4b", results_dir="results/baselines", summary_path="results/energy_summary.csv"):
    summary = load_energy_summary(summary_path)
    rows = []
    for r in load_json_dir(results_dir):          # one JSON per baseline: {"name","f1_weighted","latency_ms",...}
        rows.append({"name": r["name"], "f1": r["f1_weighted"], "latency_ms": r.get("latency_ms"),
                     "energy_mj": energy_mj(summary, platform, r["name"])})
    return rows


def print_comparison(**kw):
    for r in comparison(**kw):
        print(f"{r['name']:<18} {r['energy_mj']:8.2f} mJ  F1={r['f1']:.1f}  lat={r['latency_ms']}")
