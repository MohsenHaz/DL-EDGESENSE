"""On-device profiling."""
import torch, time, numpy as np

def profile_on_device(model, inputs, reps=100):
    model.eval()
    times = []
    with torch.no_grad():
        for _ in range(reps):
            t0 = time.perf_counter()
            model(inputs)
            times.append((time.perf_counter()-t0)*1000)
    return {"mean_ms": np.mean(times), "std_ms": np.std(times)}
