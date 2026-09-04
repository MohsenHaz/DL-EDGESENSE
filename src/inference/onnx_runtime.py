"""ONNX inference runtime."""
try:
    import onnxruntime as ort
    def run_onnx(model_path: str, input_dict: dict):
        sess = ort.InferenceSession(model_path)
        return sess.run(None, input_dict)
except ImportError:
    def run_onnx(*a, **kw): raise RuntimeError("onnxruntime not installed")
