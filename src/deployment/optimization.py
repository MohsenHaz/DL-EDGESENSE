"""Model optimization — pruning, quantization, distillation."""
import torch

def prune_model(model, sparsity: float = 0.3):
    import torch.nn.utils.prune as prune
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=sparsity)
    return model
