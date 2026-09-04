"""Federated server — 20 clients, 50 rounds (Section 4.4.8)."""
from .fed_avg import fedavg
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FederatedServer:
    def __init__(self, global_model, clients, config):
        self.model = global_model
        self.clients = clients
        self.rounds = config["federated"]["rounds"]

    def run(self, device="cpu"):
        for rnd in range(self.rounds):
            trained = [c.train(device) for c in self.clients]
            self.model = fedavg(self.model, trained)
            logger.info(f"Federated round {rnd+1}/{self.rounds} complete")
        return self.model
