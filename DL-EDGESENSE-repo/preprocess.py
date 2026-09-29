#!/usr/bin/env python3
"""
preprocess.py -- Step 3 of Supplementary S5 reproduction instructions.

Builds and saves the deterministic 70/15/15 train/val/test split
manifests for all four benchmark datasets, using the settings in
configs/training.yaml (seeds) and Table 2 / Supplementary S4.

Usage:
    python preprocess.py --dataset all --raw-root /path/to/downloaded/datasets
    python preprocess.py --dataset mmact --raw-root /path/to/mmact
"""
import argparse
import os

import yaml

from src.data.loaders.mmact_loader import MMActDataset
from src.data.loaders.other_loaders import (
    EnviroSenseDataset, SpectrumSenseDataset, DeepSense6GDataset,
)
from src.utils.logger import get_logger

log = get_logger("preprocess")

DATASETS = {
    "mmact": MMActDataset,
    "envirosense": EnviroSenseDataset,
    "spectrumsense": SpectrumSenseDataset,
    "deepsense6g": DeepSense6GDataset,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=list(DATASETS) + ["all"], default="all")
    parser.add_argument("--raw-root", required=True,
                         help="Directory containing one subfolder per dataset, "
                              "downloaded from the URLs in the manuscript's "
                              "Data Availability section.")
    parser.add_argument("--manifest-out", default="data/splits")
    parser.add_argument("--config", default="configs/training.yaml")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    seed = cfg["random_seeds"][0]

    targets = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        raw_dir = os.path.join(args.raw_root, name)
        if not os.path.isdir(raw_dir):
            log.warning(f"[{name}] raw_dir {raw_dir} does not exist -- skipping. "
                        f"Download it per Data Availability and re-run.")
            continue
        for split in ("train", "val", "test"):
            ds = DATASETS[name](raw_dir=raw_dir, split=split, seed=seed)
            out_path = os.path.join(args.manifest_out, name, f"{split}.json")
            ds.save_split_manifest(out_path)
            log.info(f"[{name}] {split}: {len(ds)} samples -> {out_path}")

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
