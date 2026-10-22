"""Model creation factory."""
from .dl_egsense import DLEgSense

def create_model(config) -> DLEgSense:
    return DLEgSense(
        num_classes=config.get("num_classes", 12),
        embed_dim=config["model"]["embedding_dim"],
        num_heads=config["model"]["num_attention_heads"],
        lambda_e=config["training"]["lambda_e"],
        lambda_c=config["training"]["lambda_c"],
    )
