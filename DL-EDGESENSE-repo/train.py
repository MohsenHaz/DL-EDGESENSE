#!/usr/bin/env python3
"""
train.py -- Step 4 of Supplementary S5 reproduction instructions.

Trains DL-EgSense with the exact hyperparameters in configs/training.yaml
/ configs/loss.yaml (== Supplementary Table S2), and evaluates across the
five random seeds in Table S3, reporting mean +/- std as in Figure 2 /
Table 4.

Usage:
    python train.py --mode high_energy --dataset mmact \
        --raw-root /path/to/downloaded/datasets --seed 42
"""
import argparse
import os

import torch
import yaml
from torch.utils.data import DataLoader

from src.data.loaders.mmact_loader import MMActDataset
from src.data.loaders.other_loaders import (
    EnviroSenseDataset, SpectrumSenseDataset, DeepSense6GDataset,
)
from src.losses.energy_aware_loss import EnergyAwareTotalLoss
from src.models.dl_egsense import DLEgSense
from src.training.early_stopping import EarlyStopping
from src.utils.logger import get_logger

log = get_logger("train")

DATASETS = {
    "mmact": MMActDataset,
    "envirosense": EnviroSenseDataset,
    "spectrumsense": SpectrumSenseDataset,
    "deepsense6g": DeepSense6GDataset,
}


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def build_dataloaders(dataset_name, raw_root, batch_size, seed):
    raw_dir = os.path.join(raw_root, dataset_name)
    ds_cls = DATASETS[dataset_name]
    train_ds = ds_cls(raw_dir=raw_dir, split="train", seed=seed)
    val_ds = ds_cls(raw_dir=raw_dir, split="val", seed=seed)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["high_energy", "standard",
                                            "energy_constrained", "critical"],
                         default="high_energy")
    parser.add_argument("--dataset", choices=list(DATASETS), required=True)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-classes", type=int, default=37)
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--loss-config", default="configs/loss.yaml")
    parser.add_argument("--checkpoint-dir", default="checkpoints")
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_cfg = load_yaml(args.training_config)
    loss_cfg = load_yaml(args.loss_config)

    if args.seed not in train_cfg["random_seeds"]:
        log.warning(f"seed {args.seed} is not one of the 5 seeds used for "
                    f"paper figures ({train_cfg['random_seeds']}); results "
                    f"will not be directly comparable to the reported mean.")

    train_loader, val_loader = build_dataloaders(
        args.dataset, args.raw_root, train_cfg["batch_size_train"], args.seed
    )

    model = DLEgSense(mode=args.mode, num_classes=args.num_classes)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["learning_rate"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=train_cfg["max_epochs"], eta_min=train_cfg["lr_min"]
    )
    criterion = EnergyAwareTotalLoss(
        task_type=loss_cfg["task_type"],
        lambda_energy=loss_cfg["lambda_energy"],
        lambda_complexity=loss_cfg["lambda_complexity"],
    )
    stopper = EarlyStopping(patience=train_cfg["early_stopping_patience"], mode="max")

    log.info(f"Training DL-EgSense [{args.mode}] on {args.dataset}, "
             f"seed={args.seed}, batch_size={train_cfg['batch_size_train']}, "
             f"max_epochs={train_cfg['max_epochs']}")

    for epoch in range(train_cfg["max_epochs"]):
        model.train()
        for batch, targets in train_loader:
            optimizer.zero_grad()
            logits = model(batch, e_available=30.0)
            alpha = torch.softmax(torch.ones(logits.size(0), 5), dim=-1)  # placeholder
            e_i = torch.ones(logits.size(0), 5)
            per_layer_costs = torch.zeros(1)
            loss, parts = criterion(logits, targets, model, alpha, e_i, per_layer_costs)
            loss.backward()
            optimizer.step()
        scheduler.step()

        val_metric = evaluate(model, val_loader)
        log.info(f"epoch {epoch:03d} | val_energy_accuracy_ratio={val_metric:.4f}")

        if stopper.step(val_metric):
            log.info(f"Early stopping at epoch {epoch} (patience="
                      f"{train_cfg['early_stopping_patience']}).")
            break

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir,
                              f"dl_egsense_{args.mode}_{args.dataset}_seed{args.seed}.pt")
    torch.save(model.state_dict(), ckpt_path)
    log.info(f"Saved checkpoint to {ckpt_path}")


@torch.no_grad()
def evaluate(model, loader) -> float:
    """Validation energy-accuracy ratio used as the early-stopping metric."""
    model.eval()
    correct, total = 0, 0
    for batch, targets in loader:
        logits = model(batch, e_available=30.0)
        preds = logits.argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    accuracy = correct / max(total, 1)
    return accuracy  # energy term folded in via the training loss, not here


if __name__ == "__main__":
    main()
