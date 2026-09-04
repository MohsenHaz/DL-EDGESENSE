# DL-EgSense: Multimodal Deep Learning for Energy-Aware Edge Perception in 6G Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange.svg)](https://pytorch.org/)

Official implementation of **DL-EgSense** from the paper:

> *"Harnessing Multimodal Deep Learning for Energy-Aware Edge Perception in 6G-Native Edge Networks"*  
> Computers, Materials & Continua, 2025

## Overview

DL-EgSense is a comprehensive deep learning framework for energy-aware multimodal sensing in 6G edge environments. It integrates three core innovations:

- **EACMA** — Energy-Adaptive Cross-Modal Attention (Section 3.5, Eq. 11)
- **HPEQ** — Hybrid Precision-Energy Quantization (Section 3.6, Eq. 12)
- **Energy-Aware NAS** — Low-power neural architecture search (Section 3.9, Eq. 19–22)

**Key results:** Up to 78% energy savings vs DeepSense, 92.7% weighted-average F1, deployed on Jetson Nano and Raspberry Pi 4B.

## Installation

```bash
git clone https://github.com/your-org/DL-EgSense.git
cd DL-EgSense
pip install -r requirements.txt
# or with conda:
conda env create -f environment.yml
conda activate dl-egsense
```

## Quick Start

```bash
# Download datasets
python scripts/download_datasets.py

# Train (standard mode)
python scripts/train.py --config configs/standard_mode.yaml

# Evaluate
python scripts/evaluate.py --config configs/standard_mode.yaml --checkpoint pretrained_weights/dl_egsense_standard.pth

# Run full benchmark
python scripts/run_benchmark.py --config configs/base_config.yaml
```

## Reproducibility

Random seeds used: `[0, 42, 123, 456, 789]`  
All results are mean ± 95% CI over 5 seeds with Bonferroni correction.  
Energy measured via INA219 @ 10 kHz, 60-second warm-up, 100 inference repetitions.

See `docs/reproduction_guide.md` for step-by-step instructions.

## Repository Structure

```
DL-EgSense/
├── configs/              # All hyperparameter configs (Table 8 modes)
├── data/                 # Dataset loading and preprocessing
├── src/
│   ├── models/           # DL-EgSense architecture
│   │   ├── eacma.py      # Energy-Adaptive Cross-Modal Attention
│   │   ├── hpeq.py       # Hybrid Precision-Energy Quantization
│   │   ├── modality_extractors/
│   │   ├── attention/
│   │   ├── fusion/
│   │   └── nas/
│   ├── training/         # Training pipeline + federated + continual
│   ├── inference/        # Inference engine + quantization
│   ├── evaluation/       # Metrics, robustness, adversarial, harvesting
│   ├── utils/            # Logging, config, reproducibility
│   └── deployment/       # RPi / Jetson Nano deployment scripts
├── scripts/              # Entry-point scripts (train, eval, ablation…)
├── notebooks/            # 10 analysis notebooks
├── tests/                # Unit tests for all components
├── pretrained_weights/   # Model checkpoints (placeholder — see below)
├── results/              # Logs, figures, tables (CSV)
└── docs/                 # Guides and API reference
```

## Pretrained Weights

Pretrained weights are hosted separately due to file size. Download via:

```bash
python scripts/download_datasets.py --weights-only
```

| Mode              | Params  | Energy  | Acc    |
|-------------------|---------|---------|--------|
| High-energy       | 846K    | 48.2 mJ | 92.7%  |
| Standard          | 584K    | 32.5 mJ | 91.8%  |
| Energy-constrained| 325K    | 18.7 mJ | 88.3%  |
| Critical          | 168K    |  9.3 mJ | 82.1%  |
| Recovery          | 412K    | 24.6 mJ | 89.5%  |

## Citation

```bibtex
@article{dl_egsense_2025,
  title   = {Harnessing Multimodal Deep Learning for Energy-Aware Edge Perception in 6G-Native Edge Networks},
  journal = {Computers, Materials \& Continua},
  year    = {2025}
}
```

## License

MIT License — see [LICENSE](LICENSE).
