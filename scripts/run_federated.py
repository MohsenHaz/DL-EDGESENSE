"""Run federated learning — 20 clients, 50 rounds (Section 4.4.8)."""
import torch
from src.utils.config_loader import load_config
from src.models.model_factory import create_model
from src.training.federated.server import FederatedServer
from src.training.federated.client import FederatedClient
from data.dataset_loader import get_dataloaders

def main():
    config = load_config("configs/federated_config.yaml")
    global_model = create_model(config)
    num_clients = config["federated"]["num_clients"]

    clients = []
    for i in range(num_clients):
        train_loader, _, _ = get_dataloaders(config)
        clients.append(FederatedClient(i, global_model, train_loader, config))

    server = FederatedServer(global_model, clients, config)
    trained_model = server.run()
    torch.save(trained_model.state_dict(), "pretrained_weights/dl_egsense_federated.pth")
    print("Federated training complete.")

if __name__ == "__main__":
    main()
