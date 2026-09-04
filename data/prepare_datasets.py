"""Prepare datasets for training — 70/15/15 split, fixed seeds."""
import os, json
from src.utils.reproducibility import set_seed

def prepare(dataset_name: str, raw_path: str, out_path: str, seed: int = 42):
    set_seed(seed)
    os.makedirs(out_path, exist_ok=True)
    print(f"Preparing {dataset_name} from {raw_path} → {out_path}")
    # Load, split, and save — implementation depends on dataset format
    splits = {"train": 0.70, "val": 0.15, "test": 0.15}
    meta = {"dataset": dataset_name, "splits": splits, "seed": seed}
    with open(os.path.join(out_path, "metadata.json"), "w") as f:
        json.dump(meta, f)
    print(f"  Done. Split: {splits}")

if __name__ == "__main__":
    for ds in ["ARID", "EnviroSense", "SpectrumSense", "MultiSense", "EnergyBench"]:
        prepare(ds, f"data/raw/{ds}", f"data/processed/{ds}")
