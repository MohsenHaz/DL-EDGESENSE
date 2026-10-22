"""Generate result tables from measured traces. Fails loudly (MissingMeasurementError) if measurements are absent.
Implemented here: Table 3 (energy), Table 8 (mode x platform), Table 17 (modelled vs measured). Table 19 comes from
scripts/run_ablation.py --phase aggregate. F1/latency columns come from results/per_run/*.json."""
import csv, os
from src.evaluation.results_io import load_energy_summary, energy_mj, load_json_dir

OUT = "paper_artifacts/tables_generated"


def write(name, header, rows):
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    print("wrote", f"{OUT}/{name}.csv")


def table03(platform="rpi4b", ref="DeepSense"):
    s = load_energy_summary(); names = sorted({r["config"] for r in s if r["platform"] == platform})
    ref_e = energy_mj(s, platform, ref); rows = []
    for n in names:
        e = energy_mj(s, platform, n); rows.append([n, f"{e:.1f}", f"{(ref_e - e) / ref_e * 100:.1f}"])
    write("table03", ["Approach", "Measured energy [mJ]", f"Savings vs {ref} [%]"], rows)


def table08():
    s = load_energy_summary(); rows = []
    for r in s:
        rows.append([r["platform"], r["config"], r["energy_mj_mean"], r["energy_mj_median"], r["latency_ms_mean"],
                     r["mean_in_inference_power_mw"], r["max_instantaneous_power_mw"]])
    write("table08", ["platform", "config", "energy_mean_mJ", "energy_median_mJ", "latency_ms",
                      "mean_in_inference_power_mW", "max_instantaneous_power_mW"], rows)


def table17():
    s = load_energy_summary(); modelled = {r["platform"]: r for r in load_json_dir("results/per_run") if "modelled_energy_mj" in r}
    rows = []
    for p, r in modelled.items():
        m = energy_mj(s, p, r["config"]); e = abs(r["modelled_energy_mj"] - m) / m * 100
        rows.append([p, f"{r['modelled_energy_mj']:.1f}", f"{m:.1f}", f"{e:.1f}"])
    if len(rows) < 3:
        print("note: fewer than 3 platforms -> no correlation coefficient is reported")
    write("table17", ["Platform", "Modelled [mJ]", "Measured [mJ]", "Error [%]"], rows)


if __name__ == "__main__":
    table03(); table08(); table17()
