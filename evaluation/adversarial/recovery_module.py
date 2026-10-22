"""Auto-recovery — reroute attention on anomaly detection (Table 18)."""
import torch

def recover(model, inputs: dict, flagged_modalities: list,
            energy_budget: torch.Tensor = None):
    """Remove flagged modalities and re-run inference."""
    clean_inputs = {k: v for k, v in inputs.items() if k not in flagged_modalities}
    with torch.no_grad():
        logits, attn = model(clean_inputs, energy_budget)
    return logits, attn
