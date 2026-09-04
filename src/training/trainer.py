"""Main training loop."""
import torch
from torch.utils.data import DataLoader
from src.utils.reproducibility import set_seed
from src.utils.logger import get_logger
from .loss_functions import EnergyAwareLoss
from .metrics_tracker import MetricsTracker

logger = get_logger(__name__)

def train(model, train_loader, val_loader, config, device="cpu"):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
    criterion = EnergyAwareLoss(lambda_e=config["training"]["lambda_e"],
                                lambda_c=config["training"]["lambda_c"])
    tracker = MetricsTracker()
    best_f1, patience_cnt = 0.0, 0

    for epoch in range(config["training"]["epochs"]):
        model.train()
        for batch in train_loader:
            inputs, targets = batch["inputs"], batch["labels"].to(device)
            energy_budget    = batch.get("energy_budget", torch.ones(targets.size(0), 1)).to(device)
            modality_energies= batch.get("modality_energies", torch.ones(targets.size(0), 5)).to(device)

            logits, attn = model(inputs, energy_budget, modality_energies)
            alpha = attn.mean(dim=(1, 2))  # aggregate over heads and queries
            loss  = criterion(logits, targets, alpha, modality_energies, model)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        scheduler.step()
        val_metrics = evaluate_epoch(model, val_loader, device)
        tracker.update(epoch, val_metrics)
        logger.info(f"Epoch {epoch+1}: F1={val_metrics['f1_weighted']:.2f}%")

        if val_metrics["f1_weighted"] > best_f1:
            best_f1 = val_metrics["f1_weighted"]
            torch.save(model.state_dict(), "results/checkpoints/best_model.pth")
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= config["training"]["early_stopping_patience"]:
                logger.info("Early stopping triggered.")
                break

    return tracker


def evaluate_epoch(model, loader, device):
    import numpy as np
    from src.utils.metrics_calculator import compute_metrics
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for batch in loader:
            inputs, targets = batch["inputs"], batch["labels"].to(device)
            logits, _ = model(inputs)
            all_preds.append(logits.argmax(1).cpu().numpy())
            all_tgts.append(targets.cpu().numpy())
    return compute_metrics(np.concatenate(all_tgts), np.concatenate(all_preds))
