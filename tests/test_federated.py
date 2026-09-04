"""Federated learning tests."""
import torch
from src.models.dl_egsense import DLEgSense
from src.training.federated.fed_avg import fedavg

def test_fedavg_preserves_shape():
    global_model = DLEgSense(num_classes=12, embed_dim=32)
    clients = [DLEgSense(num_classes=12, embed_dim=32) for _ in range(3)]
    agg = fedavg(global_model, clients)
    for p_g, p_c in zip(agg.parameters(), global_model.parameters()):
        assert p_g.shape == p_c.shape
