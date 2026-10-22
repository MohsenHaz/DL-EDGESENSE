"""Optimizer configuration."""
import torch.optim as optim

def get_optimizer(model, config):
    return optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
