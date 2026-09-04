"""TFLite inference runtime."""
try:
    import tensorflow as tf
    def run_tflite(model_path: str, inputs):
        interp = tf.lite.Interpreter(model_path=model_path)
        interp.allocate_tensors()
        ii = interp.get_input_details()
        oi = interp.get_output_details()
        for i, inp in enumerate(inputs):
            interp.set_tensor(ii[i]["index"], inp)
        interp.invoke()
        return interp.get_tensor(oi[0]["index"])
except ImportError:
    def run_tflite(*a, **kw): raise RuntimeError("TensorFlow not installed")
