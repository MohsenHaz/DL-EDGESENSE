"""Export to ONNX."""
import torch

def export_onnx(model, output_path="model.onnx", input_shape=(1, 3, 224, 224)):
    dummy = torch.randn(*input_shape)
    torch.onnx.export(model, dummy, output_path,
                      input_names=["visual"], output_names=["logits"],
                      dynamic_axes={"visual": {0: "batch"}})
    print(f"Exported: {output_path}")
