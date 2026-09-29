"""
Base multimodal dataset loader.

Applies the preprocessing pipeline in Supplementary Section S4 and the
70/15/15 train/val/test split (Table 2 / Table S3) identically across
all four benchmark datasets, so that a single training/evaluation loop
in train.py works unmodified across MMAct, EnviroSense, SpectrumSense,
and DeepSense 6G.

NOTE: this loader defines the *interface* and preprocessing contract.
Each dataset-specific file below points `raw_dir` at the actual
downloaded data (see Data Availability in the manuscript) and only
needs to implement `_index_samples()` and `_load_raw(sample_id)`.
"""
import json
import os
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset

SPLIT_RATIOS = (0.70, 0.15, 0.15)  # train / val / test


class BaseMultimodalDataset(Dataset):
    def __init__(self, raw_dir: str, split: str = "train", seed: int = 42,
                 transform=None):
        assert split in ("train", "val", "test")
        self.raw_dir = Path(raw_dir)
        self.split = split
        self.transform = transform
        self.samples = self._deterministic_split(self._index_samples(), seed)

    def _deterministic_split(self, all_ids, seed):
        rng = random.Random(seed)
        ids = list(all_ids)
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(n * SPLIT_RATIOS[0])
        n_val = int(n * SPLIT_RATIOS[1])
        if self.split == "train":
            return ids[:n_train]
        elif self.split == "val":
            return ids[n_train:n_train + n_val]
        else:
            return ids[n_train + n_val:]

    def _index_samples(self):
        """Return a list of sample identifiers. Override per dataset."""
        raise NotImplementedError

    def _load_raw(self, sample_id):
        """Return (modality_dict, label). Override per dataset."""
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_id = self.samples[idx]
        modality_dict, label = self._load_raw(sample_id)
        if self.transform is not None:
            modality_dict = self.transform(modality_dict)
        return modality_dict, label

    def save_split_manifest(self, out_path: str):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w") as f:
            json.dump({"split": self.split, "sample_ids": self.samples}, f, indent=2)
