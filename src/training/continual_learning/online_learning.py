"""Online adaptation for deployment."""
import torch

def online_update(model, batch, optimizer, criterion, ewc=None):
    model.train()
    inputs, targets = batch["inputs"], batch["labels"]
    logits, _ = model(inputs)
    loss = criterion(logits, targets)
    if ewc:
        loss += ewc.penalty(model)
    optimizer.zero_grad(); loss.backward(); optimizer.step()
    return loss.item()
