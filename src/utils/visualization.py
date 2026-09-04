"""Visualization utilities."""
import matplotlib.pyplot as plt
import numpy as np

def plot_energy_accuracy_tradeoff(energies, accuracies, labels, save_path=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    for e, a, l in zip(energies, accuracies, labels):
        ax.scatter(e, a, label=l, s=100)
    ax.set_xlabel("Energy (mJ)"); ax.set_ylabel("F1 Score (%)")
    ax.set_title("Energy-Accuracy Tradeoff (Figure 4)")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    return fig
