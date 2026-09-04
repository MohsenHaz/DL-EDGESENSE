# DL-EgSense Architecture

## Core components

| Component | File | Paper reference |
|-----------|------|----------------|
| EACMA     | src/models/eacma.py | Section 3.5, Eq 11 |
| HPEQ      | src/models/hpeq.py  | Section 3.6, Eq 12 |
| Energy-Aware Loss | src/training/loss_functions.py | Section 3.8, Eq 15-18 |
| NAS | src/models/nas/energy_aware_nas.py | Section 3.9, Eq 19-22 |
| EWC | src/training/continual_learning/ewc.py | Section 4.4.10 |

## Data flow
1. Raw sensor data → modality extractors → feature vectors (B, D)
2. Stack → (B, N_mod, D)
3. EACMA cross-modal attention → energy-weighted fused features
4. Adaptive fusion (early/hybrid/late per Eq 13) → pooled (B, D)
5. HPEQ classifier → logits (B, num_classes)
