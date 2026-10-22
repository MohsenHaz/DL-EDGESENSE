"""End-to-end latency analysis — Table 6."""
import time, numpy as np, torch

def measure_latency(model, input_sample, reps=100, device="cpu"):
    model.eval().to(device)
    times = []
    with torch.no_grad():
        for _ in range(reps):
            t0 = time.perf_counter()
            model(input_sample)
            times.append((time.perf_counter() - t0) * 1000)
    return {"mean_ms": np.mean(times), "std_ms": np.std(times), "p95_ms": np.percentile(times, 95)}
