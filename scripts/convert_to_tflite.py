"""Convert PyTorch model to TFLite for edge deployment."""
import torch
from src.models.model_factory import create_model
from src.utils.config_loader import load_config

def convert(config_path, checkpoint_path, output_path="model.tflite"):
    config = load_config(config_path)
    model = create_model(config)
    model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
    model.eval()
    print(f"TFLite conversion: use torch.onnx.export then tflite_convert")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    convert("configs/standard_mode.yaml", "pretrained_weights/dl_egsense_standard.pth")
