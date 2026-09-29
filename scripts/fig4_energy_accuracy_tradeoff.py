import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "Nimbus Roman", "DejaVu Serif"]
rcParams["font.size"] = 9
rcParams["axes.linewidth"] = 0.8

out_dir = os.path.join(os.path.expanduser("~"), "Downloads", "DL-EgSense")
os.makedirs(out_dir, exist_ok=True)

# DL-EgSense operating modes, Table 12
modes = ["High-Energy", "Standard", "Recovery", "Energy-Constrained", "Critical"]
mode_energy = [51.0, 32.5, 24.6, 18.7, 9.3]
mode_acc = [92.7, 91.8, 89.5, 88.3, 82.1]

# Baselines, Table 4 (single operating point each)
baselines = [
    ("MobileNetV3 [25]", 73.4, 89.2, "s"),
    ("EfficientNet-B0 [26]", 89.7, 91.5, "^"),
    ("EdgeBERT [48]", 67.2, 88.3, "v"),
    ("MicroNet [49]", 82.1, 90.1, "D"),
    ("DeepSense [38]", 146.0, 91.5, "P"),
    ("FBNet-C [50]", 78.9, 89.7, "X"),
    ("ProxylessNAS [51]", 85.3, 90.8, "<"),
    ("Once-for-All [30]", 91.2, 91.2, ">"),
]

fig, ax = plt.subplots(figsize=(3.5, 3.2))
ax.plot(mode_energy, mode_acc, "-o", color="#1f4e79", lw=1.2, ms=4.5,
        label="DL-EgSense (Table 12 modes)", zorder=3)
for m, e, a in zip(modes, mode_energy, mode_acc):
    ax.annotate(m, (e, a), textcoords="offset points", xytext=(5, -3 if m in ("Critical","High-Energy") else -9),
                fontsize=6, color="#1f4e79")
for name, e, a, mk in baselines:
    ax.scatter(e, a, marker=mk, s=28, label=name, zorder=3, edgecolors="black", linewidths=0.4)

ax.set_xlabel("Energy per inference (mJ)")
ax.set_ylabel("Accuracy (%)")
ax.set_xlim(0, 160)
ax.set_ylim(80, 94)
ax.grid(True, ls=":", lw=0.5, alpha=0.6)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False, handletextpad=0.3, columnspacing=0.6)
fig.tight_layout()
fig.savefig(os.path.join(out_dir, "fig4_energy_accuracy_tradeoff.png"), dpi=600)
fig.savefig(os.path.join(out_dir, "fig4_energy_accuracy_tradeoff.pdf"))
