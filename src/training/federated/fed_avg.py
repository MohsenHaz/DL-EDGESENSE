"""FedAvg aggregation — Section 4.4.8."""
import torch, copy

def fedavg(global_model, client_models, client_weights=None):
    """Aggregate client models into global model."""
    n = len(client_models)
    w = client_weights or [1.0 / n] * n
    global_state = copy.deepcopy(global_model.state_dict())
    for key in global_state:
        global_state[key] = sum(w[i] * client_models[i].state_dict()[key]
                                 for i in range(n))
    global_model.load_state_dict(global_state)
    return global_model
