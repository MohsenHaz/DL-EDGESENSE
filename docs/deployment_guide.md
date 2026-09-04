# Edge Deployment Guide

## Raspberry Pi 4B
```python
from src.deployment.raspberry_pi import deploy_rpi
engine = deploy_rpi("pretrained_weights/dl_egsense_standard.pth", mode="standard")
result = engine.infer(inputs)
```

## NVIDIA Jetson Nano
```python
from src.deployment.jetson_nano import deploy_jetson
engine = deploy_jetson("pretrained_weights/dl_egsense_high_energy.pth", mode="high_energy")
```

## Energy measurement
- INA219 current sensor @ 10 kHz sampling rate
- 60-second thermal warm-up before each session
- 100 inference repetitions per configuration
- Idle power measured separately and subtracted
