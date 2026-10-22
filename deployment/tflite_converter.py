"""Convert to TFLite for edge inference."""
def convert_to_tflite(model_path: str, output_path: str = "model.tflite"):
    print(f"Converting {model_path} → {output_path}")
    print("Use: torch.onnx.export → onnx-tf → tflite_convert")
