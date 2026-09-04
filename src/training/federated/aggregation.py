"""Aggregation strategies (FedAvg, weighted, etc.)."""
from .fed_avg import fedavg

AGGREGATION = {"fedavg": fedavg}

def get_aggregation(name: str):
    return AGGREGATION[name]
