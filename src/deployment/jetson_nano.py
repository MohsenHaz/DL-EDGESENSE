"""NVIDIA Jetson Nano deployment — 128-core Maxwell GPU (Table 8)."""
import torch
from src.inference.inference_engine import InferenceEngine
from src.models.model_factory import create_model
from src.utils.config_loader import load_config

def deploy_jetson(checkpoint_path: str, mode: str = "high_energy"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = load_config(f"configs/{mode}_mode.yaml")
    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    engine = InferenceEngine(model, device=device)
    print(f"DL-EgSense deployed on Jetson Nano — mode={mode}, device={device}")
    return engine
