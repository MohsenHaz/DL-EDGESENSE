"""Federated client — local training for E local epochs."""
import torch, copy

class FederatedClient:
    def __init__(self, client_id, model, data_loader, config):
        self.id = client_id
        self.model = copy.deepcopy(model)
        self.loader = data_loader
        self.epochs = config["federated"]["local_epochs"]

    def train(self, device="cpu"):
        self.model.to(device).train()
        opt = torch.optim.Adam(self.model.parameters(), lr=0.001)
        loss_fn = torch.nn.CrossEntropyLoss()
        for _ in range(self.epochs):
            for batch in self.loader:
                inputs, targets = batch["inputs"], batch["labels"].to(device)
                logits, _ = self.model(inputs)
                loss = loss_fn(logits, targets)
                opt.zero_grad(); loss.backward(); opt.step()
        return self.model
