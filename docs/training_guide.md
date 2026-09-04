# Training Guide

## Hyperparameters (Table 8 / base_config.yaml)

| Parameter | Value | Description |
|-----------|-------|-------------|
| lambda_e | 1.0 | Energy loss weight |
| lambda_c | 0.1 | Complexity loss weight |
| k | 10.0 | Sigmoid sharpness (Eq 18) |
| tau | 0.5 | Sigmoid threshold (Eq 18) |
| Seeds | 0,42,123,456,789 | Fixed for reproducibility |
| batch_size | 64 | Training batch size |
| lr | 0.001 | Adam learning rate |

## Energy loss formulation (Equations 15-18)
See src/training/loss_functions.py
