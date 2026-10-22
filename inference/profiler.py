"""Performance profiling."""
import torch, time

def profile_model(model, input_sample, device="cpu", reps=100):
    model.eval().to(device)
    latencies = []
    with torch.no_grad():
        for _ in range(reps):
            t0 = time.perf_counter()
            model(input_sample)
            latencies.append((time.perf_counter() - t0) * 1000)
    import numpy as np
    return {"mean_ms": np.mean(latencies), "std_ms": np.std(latencies),
            "p95_ms": np.percentile(latencies, 95)}
