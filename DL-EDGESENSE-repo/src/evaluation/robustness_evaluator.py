"""Robustness evaluation — Table 14 (temporal stability, 14-day window)."""
import numpy as np

def temporal_stability(model, daily_loaders, device="cpu"):
    """Evaluate model across 14 days; return per-day F1."""
    from .metrics import evaluate
    import torch
    results = []
    for day, loader in enumerate(daily_loaders):
        preds, tgts = [], []
        model.eval()
        with torch.no_grad():
            for batch in loader:
                logits, _ = model(batch["inputs"])
                preds.extend(logits.argmax(1).cpu().tolist())
                tgts.extend(batch["labels"].tolist())
        results.append({"day": day+1, **evaluate(tgts, preds)})
    return results
