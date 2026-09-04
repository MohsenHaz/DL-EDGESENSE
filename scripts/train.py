"""Main training script — python scripts/train.py --config configs/standard_mode.yaml"""
import argparse, torch
from src.utils.config_loader import load_config
from src.utils.reproducibility import set_seed, SEEDS
from src.models.model_factory import create_model
from src.training.trainer import train
from data.dataset_loader import get_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard_mode.yaml")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = [args.seed] if args.seed else SEEDS

    all_results = []
    for seed in seeds:
        set_seed(seed)
        model = create_model(config)
        train_loader, val_loader, _ = get_dataloaders(config)
        tracker = train(model, train_loader, val_loader, config, device)
        all_results.append(tracker.best())

    # Report mean ± CI across seeds
    import numpy as np
    f1s = [r["f1_weighted"] for r in all_results]
    print(f"F1 (weighted): {np.mean(f1s):.2f} ± {1.96*np.std(f1s)/np.sqrt(len(f1s)):.2f}% (95% CI)")

if __name__ == "__main__":
    main()
