"""Checkpoint management."""
import torch, os

def save(model, optimizer, epoch, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({"epoch": epoch, "model": model.state_dict(),
                "optimizer": optimizer.state_dict()}, path)

def load(model, optimizer, path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt["epoch"]
