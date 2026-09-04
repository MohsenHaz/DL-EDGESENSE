"""Utility functions for data handling."""
import torch

def collate_multimodal(batch):
    """Custom collate for variable modality dicts."""
    labels = torch.stack([b["labels"] for b in batch])
    modalities = set().union(*[b["inputs"].keys() for b in batch])
    inputs = {m: torch.stack([b["inputs"][m] for b in batch if m in b["inputs"]])
              for m in modalities}
    return {"inputs": inputs, "labels": labels}
