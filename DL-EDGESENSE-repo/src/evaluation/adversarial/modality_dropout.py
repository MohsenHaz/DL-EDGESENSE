"""Modality dropout simulation — deactivate one or more modalities at runtime."""
def dropout_modality(inputs: dict, modality: str) -> dict:
    return {k: v for k, v in inputs.items() if k != modality}

def random_dropout(inputs: dict, p: float = 0.3) -> dict:
    import random
    return {k: v for k, v in inputs.items() if random.random() > p}
