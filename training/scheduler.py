"""Learning rate schedulers."""
import torch.optim.lr_scheduler as sched

def get_scheduler(optimizer, config):
    return sched.CosineAnnealingLR(optimizer, T_max=config["training"]["epochs"])
