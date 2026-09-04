"""Load ARID, EnviroSense, SpectrumSense, MultiSense, Energy-Bench datasets."""
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class MultimodalDataset(Dataset):
    """Generic multimodal dataset wrapper."""
    def __init__(self, data_path: str, modalities: list, split: str = "train"):
        self.modalities = modalities
        self.split = split
        # Placeholder — replace with actual dataset loading per dataset_configs.yaml
        self.length = {"train": 700, "val": 150, "test": 150}[split]

    def __len__(self): return self.length

    def __getitem__(self, idx):
        sample = {"labels": torch.randint(0, 12, (1,)).squeeze()}
        sample["energy_budget"] = torch.rand(1)
        sample["modality_energies"] = torch.rand(len(self.modalities)) * 0.1 + 0.01
        sample["inputs"] = {}
        if "visual" in self.modalities:
            sample["inputs"]["visual"] = torch.randn(3, 224, 224)
        if "acoustic" in self.modalities:
            sample["inputs"]["acoustic"] = torch.randn(1, 64, 128)  # mel-spectrogram
        if "imu" in self.modalities:
            sample["inputs"]["imu"] = torch.randn(100, 9)           # T x 9-axis
        if "environmental" in self.modalities:
            sample["inputs"]["environmental"] = torch.randn(3)
        if "rf" in self.modalities:
            sample["inputs"]["rf"] = torch.randn(512)
        return sample


def get_dataloaders(config, dataset_name="MultiSense", batch_size=64):
    modalities = config["datasets"][dataset_name]["modalities"]
    train_ds = MultimodalDataset("", modalities, "train")
    val_ds   = MultimodalDataset("", modalities, "val")
    test_ds  = MultimodalDataset("", modalities, "test")
    kw = dict(batch_size=batch_size, num_workers=2, pin_memory=True)
    return (DataLoader(train_ds, shuffle=True, **kw),
            DataLoader(val_ds, **kw),
            DataLoader(test_ds, **kw))
