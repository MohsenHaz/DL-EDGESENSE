# Step-by-Step Reproduction Guide

## 1. Environment setup
```bash
conda env create -f environment.yml
conda activate dl-egsense
```

## 2. Download datasets (Table 2)
```bash
python scripts/download_datasets.py
python data/prepare_datasets.py
```

## 3. Train with 5 seeds (0, 42, 123, 456, 789)
```bash
for seed in 0 42 123 456 789; do
    python scripts/train.py --config configs/standard_mode.yaml --seed $seed
done
```

## 4. Evaluate
```bash
python scripts/evaluate.py --config configs/standard_mode.yaml \
    --checkpoint pretrained_weights/dl_egsense_standard.pth
```

## 5. Run ablation study (Table 19)
```bash
python scripts/run_ablation.py
```

## 6. Run benchmark vs baselines (Tables 3-4)
```bash
python scripts/run_benchmark.py
```

## 7. Adversarial evaluation (Table 18)
```bash
python scripts/run_adversarial.py
```

## 8. Federated learning (Section 4.4.8)
```bash
python scripts/run_federated.py
```

## Expected results
- F1 (weighted): 92.7% ± 0.3% (95% CI, 5 seeds)
- Energy (standard mode): 51.0 mJ ± 1.2 mJ
- Latency (Jetson Nano): 115 ms
