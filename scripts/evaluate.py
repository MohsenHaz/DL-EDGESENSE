"""Main evaluation script."""
import argparse, torch, numpy as np
from src.utils.config_loader import load_config
from src.models.model_factory import create_model
from src.evaluation.metrics import evaluate
from src.evaluation.energy_profiler import profile_all_modes
from data.dataset_loader import get_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/standard_mode.yaml")
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = create_model(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval().to(device)

    _, _, test_loader = get_dataloaders(config)
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for batch in test_loader:
            logits, _ = model(batch["inputs"])
            all_preds.extend(logits.argmax(1).cpu().tolist())
            all_tgts.extend(batch["labels"].tolist())

    metrics = evaluate(all_tgts, all_preds)
    print("Evaluation results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

if __name__ == "__main__":
    main()
