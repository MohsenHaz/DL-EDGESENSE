"""Run continual learning — EWC over 14-day deployment window."""
import torch
from src.models.model_factory import create_model
from src.utils.config_loader import load_config
from src.training.continual_learning.ewc import EWC
from src.training.continual_learning.online_learning import online_update
from data.dataset_loader import get_dataloaders

def main():
    config = load_config("configs/standard_mode.yaml")
    model = create_model(config)
    train_loader, _, _ = get_dataloaders(config)
    ewc = EWC(model, train_loader)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    print("Continual learning with EWC initialised.")
    # Day-by-day online updates would run here

if __name__ == "__main__":
    main()
