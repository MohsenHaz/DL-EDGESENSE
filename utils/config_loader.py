"""Load YAML config files with OmegaConf."""
from omegaconf import OmegaConf
from pathlib import Path

def load_config(path: str):
    return OmegaConf.load(path)

def merge_configs(base_path: str, override_path: str):
    base = OmegaConf.load(base_path)
    override = OmegaConf.load(override_path)
    return OmegaConf.merge(base, override)
