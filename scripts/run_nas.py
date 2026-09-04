"""Run Energy-Aware NAS — Section 3.9."""
import torch
from src.models.nas.energy_aware_nas import EnergyAwareNAS

def main():
    nas = EnergyAwareNAS(lambda_energy=0.3)
    arch_optimizer = torch.optim.Adam(nas.parameters(), lr=0.003)

    for step in range(50):
        arch_optimizer.zero_grad()
        loss = nas.arch_loss()
        loss.backward()
        arch_optimizer.step()
        if step % 10 == 0:
            print(f"Step {step}: arch_loss={loss.item():.4f}")

    best = nas.get_best_arch()
    print(f"Best architecture: {best}")

if __name__ == "__main__":
    main()
