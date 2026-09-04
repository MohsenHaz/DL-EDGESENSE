"""Generate all paper figures using matplotlib."""
import matplotlib.pyplot as plt
import numpy as np
import os; os.makedirs("results/figures", exist_ok=True)

def figure_energy_comparison():
    models  = ["DeepSense", "EdgeML", "Fed-Sense", "EffSensing", "DL-EgSense"]
    energy  = [146, 128.2, 138.7, 111.0, 51.0]
    colors  = ["#aaa"]*4 + ["#1f6b3a"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(models, energy, color=colors)
    ax.set_ylabel("Energy per inference (mJ)")
    ax.set_title("Energy Consumption Comparison (Figure 2)")
    fig.savefig("results/figures/figure_2_energy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

def figure_energy_accuracy_tradeoff():
    e = [146, 128.2, 138.7, 111.0, 51.0]
    f = [91.5, 75.3, 84.5, 89.7, 92.7]
    labels = ["DeepSense", "EdgeML", "Fed-Sense", "EffSensing", "DL-EgSense"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for ei, fi, l in zip(e, f, labels):
        ax.scatter(ei, fi, s=120, label=l)
    ax.set_xlabel("Energy (mJ)"); ax.set_ylabel("F1 (%)")
    ax.set_title("Energy-Accuracy Tradeoff (Figure 4)")
    ax.legend(); fig.savefig("results/figures/figure_4_energy_accuracy_tradeoff.png",
                             dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    figure_energy_comparison()
    figure_energy_accuracy_tradeoff()
    print("Figures saved to results/figures/")
