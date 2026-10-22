"""Table 19 - ablation study. Every variant is built, evaluated and (on hardware) measured; nothing is assigned.

Phase `run`      : evaluate one variant on the test split -> results/ablations/<variant>_seed<seed>.json
                   (F1, latency, parameter count). Energy is recorded separately with scripts/run_energy_profiling.py
                   style sessions listed in the MANIFEST (config = variant name).
Phase `aggregate`: combine ablation results with results/energy_summary.csv -> paper_artifacts/tables_generated/table19.csv
"""
import argparse, json, os, time, torch
import numpy as np
from src.utils.config_loader import load_config
from src.models.model_factory import create_model
from src.models.ablation import VARIANTS, apply_ablation, active_modalities
from src.utils.reproducibility import set_seed
from src.evaluation.metrics import evaluate
from src.evaluation.results_io import load_energy_summary, energy_mj, load_json_dir
from data.dataset_loader import get_dataloaders


def run(variant, config_path, checkpoint, seed, platform):
    cfg = load_config(config_path); set_seed(seed)
    model = create_model(cfg)
    model, _ = apply_ablation(model, variant, cfg)
    if checkpoint:
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=False)
    model.eval()
    _, _, test = get_dataloaders(cfg)
    preds, tgts, lat = [], [], []
    with torch.no_grad():
        for b in test:
            x = active_modalities(model, b["inputs"], variant)
            t0 = time.perf_counter(); logits, _ = model(x); lat.append((time.perf_counter() - t0) * 1e3)
            preds += logits.argmax(1).tolist(); tgts += b["labels"].tolist()
    res = {"variant": variant, "seed": seed, "platform": platform, "params": model.count_params(),
           "latency_ms_mean": float(np.mean(lat)), **evaluate(tgts, preds)}
    os.makedirs("results/ablations", exist_ok=True)
    json.dump(res, open(f"results/ablations/{variant}_seed{seed}.json", "w"), indent=1)
    return res


def aggregate(platform, out="paper_artifacts/tables_generated/table19.csv"):
    summ = load_energy_summary()
    full = energy_mj(summ, platform, "full")
    res = load_json_dir("results/ablations")
    rows = []
    for v in VARIANTS:
        f1 = [r["f1_weighted"] for r in res if r["variant"] == v]
        if not f1:
            raise SystemExit(f"missing ablation results for {v}")
        e = energy_mj(summ, platform, v)
        rows.append({"variant": v, "f1_mean": np.mean(f1), "energy_mj": e,
                     "energy_saved_pct_by_full_model": (e - full) / e * 100,       # (E_ablated - E_full) / E_ablated
                     "energy_change_pct_of_variant_vs_full": (full - e) / full * 100})  # negative = variant uses more
    os.makedirs(os.path.dirname(out), exist_ok=True)
    import csv
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["run", "aggregate"], required=True)
    ap.add_argument("--variant", choices=VARIANTS); ap.add_argument("--config", default="configs/standard_mode.yaml")
    ap.add_argument("--checkpoint"); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--platform", default="rpi4b")
    a = ap.parse_args()
    run(a.variant, a.config, a.checkpoint, a.seed, a.platform) if a.phase == "run" else aggregate(a.platform)
