"""Regenerate manuscript figures 2, 4, 5, 6, 7, 8, 9, 13 from table CSVs - no numeric literals.

--source reported  : paper_artifacts/tables_as_reported/ (numbers exactly as printed in the manuscript tables)
--source generated : paper_artifacts/tables_generated/   (produced by generate_tables.py from measured traces)
Only the retained, verifiable baselines (DeepSense, EdgeML, MobileNetV3, EfficientNet-B0, Once-for-All, ...) appear.
The withdrawn systems (EfficientSensing, Federated-Sense, MultiModal-Net) are rejected by _guard().
"""
import argparse, csv, os, re
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

WITHDRAWN = ("EfficientSensing", "Federated-Sense", "MultiModal-Net", "MultiModalNet", "Lite-computing")
COL = {"DL-EgSense": "#1f77b4", "DeepSense": "#ff7f0e", "EdgeML": "#d62728", "Once-for-All": "#2ca02c"}


def num(s):
    m = re.search(r"-?\d+(?:\.\d+)?", s.replace("–", "-").replace("−", "-"))
    return float(m.group()) if m else float("nan")


def read(src, name):
    with open(os.path.join(src, name + ".csv"), newline="") as f:
        rows = list(csv.reader(f))
    for r in rows:
        if r and any(w.lower() in r[0].lower() for w in WITHDRAWN):
            raise SystemExit(f"{name}: withdrawn system '{r[0]}' present")
    return rows


def clean(n):
    return re.sub(r"\s*\[\d+\]", "", n).strip()


def save(fig, out, name):
    fig.tight_layout(); fig.savefig(f"{out}/{name}.png", dpi=200); plt.close(fig)


def fig2(src, out):
    r = read(src, "table03")[1:]
    names = [clean(x[0]) for x in r]; e = [num(x[1]) for x in r]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.bar(names, e, color=[COL.get(n, "#7f7f7f") for n in names])
    for b, v in zip(bars, e): ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:g}", ha="center", fontsize=8)
    ax.set_ylabel("Energy per inference [mJ]"); ax.set_title("Per-inference energy, Raspberry Pi 4B (Table 3)")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right"); save(fig, out, "Fig02_Energy_Consumption_Comparison")


def fig4(src, out):
    t4 = read(src, "table04")[1:]; t12 = read(src, "table12")[1:]
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for x in t4:
        n = clean(x[0])
        if n == "DL-EgSense": continue
        ax.scatter(num(x[1]), num(x[2]), s=55, color="#7f7f7f"); ax.annotate(n, (num(x[1]), num(x[2])), fontsize=7, xytext=(3, 3), textcoords="offset points")
    e = [num(x[4]) for x in t12]; a = [num(x[5]) for x in t12]
    ax.plot(*zip(*sorted(zip(e, a))), "o-", color=COL["DL-EgSense"], label="DL-EgSense operating modes (Table 12)")
    for x in t12: ax.annotate(x[0].replace(" Mode", ""), (num(x[4]), num(x[5])), fontsize=7, xytext=(3, -10), textcoords="offset points")
    ax.set_xlabel("Energy per inference [mJ]"); ax.set_ylabel("Accuracy [%]"); ax.legend(fontsize=8); ax.grid(alpha=.3)
    save(fig, out, "Fig04_Energy_Accuracy_Tradeoff")


def fig5(src, out):
    r = read(src, "table05")[2:]; names = [x[0] for x in r]
    fig, axs = plt.subplots(1, 3, figsize=(11, 3.4))
    for k, (ttl, i) in enumerate([("Inference time [ms]", 1), ("Memory [MB]", 3), ("Throughput [inf/s]", 5)]):
        w = 0.38; x = np.arange(len(names))
        axs[k].bar(x - w / 2, [num(q[i]) for q in r], w, label="RPi 4B"); axs[k].bar(x + w / 2, [num(q[i + 1]) for q in r], w, label="Jetson Nano")
        axs[k].set_xticks(x); axs[k].set_xticklabels(names, fontsize=8); axs[k].set_title(ttl, fontsize=9)
    axs[0].legend(fontsize=8); save(fig, out, "Fig05_Computational_Overhead")


def fig6(src, out):
    r = read(src, "table06"); hdr = r[0][1:]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for x in r[1:]: ax.plot(hdr, [num(v) for v in x[1:]], "o-", label=x[0], color=COL.get(x[0]))
    ax.set_xlabel("Computational load"); ax.set_ylabel("End-to-end latency [ms]"); ax.legend(); ax.grid(alpha=.3)
    save(fig, out, "Fig06_End_to_End_Latency")


def fig7(src, out):
    r = read(src, "table07"); names = [x[0] for x in r[1:]]
    fig, ax = plt.subplots(figsize=(7, 4)); x = np.arange(len(names)); w = 0.38
    ax.bar(x - w / 2, [num(q[1]) for q in r[1:]], w, yerr=[num(q[3]) for q in r[1:]], capsize=3, label="Mean power (error bar: std dev)")
    ax.bar(x + w / 2, [num(q[2]) for q in r[1:]], w, label="Peak of windowed mean")
    ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("Power [mW]"); ax.set_title("24-hour power statistics (Table 7)"); ax.legend(fontsize=8)
    save(fig, out, "Fig07_24h_Energy_Metrics")


def fig8(src, out):
    r = read(src, "table10")[1:]; names = [x[0] for x in r]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, (ttl, i, u) in zip(axs, [("Energy reduction achieved [%]", 2, ""), ("Accuracy preserved [%]", 3, ""), ("Adaptation quality", 5, "")]):
        ax.bar(names, [num(x[i]) for x in r], color=[COL.get(n) for n in names]); ax.set_title(ttl, fontsize=9)
    save(fig, out, "Fig08_Sudden_Energy_Constraint_Response")


def fig9(src, out):
    r = read(src, "table12"); names = [x[0] for x in r[1:]]; x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.2
    for k, (lbl, i) in enumerate([("Parameters [K]", 1), ("FLOPs [M]", 2), ("Memory [MB]", 3), ("Energy [mJ]", 4)]):
        ax.bar(x + (k - 1.5) * w, [num(q[i]) for q in r[1:]], w, label=lbl)
    ax2 = ax.twinx(); ax2.plot(x, [num(q[5]) for q in r[1:]], "k--o", label="Accuracy [%]"); ax2.set_ylabel("Accuracy [%]")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=8); ax.set_ylabel("Resource metrics"); ax.legend(fontsize=7, loc="upper right")
    save(fig, out, "Fig09_Model_Complexity_Adaptation")


def fig13(src, out):
    r = read(src, "fig13_budget25")[1:]
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([x[0] for x in r], [num(x[1]) for x in r], color=[COL.get(x[0]) for x in r])
    for i, x in enumerate(r): ax.text(i, num(x[1]) + 1, x[1], ha="center", fontsize=9)
    ax.set_ylabel("Accuracy at 25% of full energy budget [%]"); ax.set_ylim(0, 100)
    save(fig, out, "Fig13_Adaptability_Dynamic_Scenarios")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--source", choices=["reported", "generated"], default="reported")
    ap.add_argument("--out", default="paper_artifacts/figures"); a = ap.parse_args()
    src = f"paper_artifacts/tables_{'as_reported' if a.source == 'reported' else 'generated'}"
    os.makedirs(a.out, exist_ok=True)
    for f in (fig2, fig4, fig5, fig6, fig7, fig8, fig9, fig13): f(src, a.out)
    print("figures written to", a.out)
