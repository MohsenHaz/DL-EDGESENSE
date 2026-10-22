"""Raspberry Pi 4B deployment — Cortex-A72, 4GB RAM (Table 8)."""
import torch
from src.inference.inference_engine import InferenceEngine
from src.models.model_factory import create_model
from src.utils.config_loader import load_config

def deploy_rpi(checkpoint_path: str, mode: str = "standard"):
    config = load_config(f"configs/{mode}_mode.yaml")
    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    engine = InferenceEngine(model, device="cpu")
    print(f"DL-EgSense deployed on RPi 4B — mode={mode}")
    return engine
