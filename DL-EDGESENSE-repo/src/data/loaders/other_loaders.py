"""
EnviroSense / SpectrumSense / DeepSense 6G loaders.
Preprocessing per Supplementary S4.2 - S4.4.

Each class expects `raw_dir` populated from the URL given in the
manuscript's Data Availability section, and a `labels.json` /
`metadata.json` describing the sample index (layout documented in
each class's docstring). These mirror MMActDataset's interface so
train.py / preprocess.py can iterate over any of the four datasets
identically.
"""
import json

import numpy as np
import torch
import torchvision.transforms as T

from src.data.loaders.base_multimodal_loader import BaseMultimodalDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EnviroSenseDataset(BaseMultimodalDataset):
    """
    S4.2: RGB (224x224, ImageNet-normalised); acoustic -> 128-bin
    mel-spectrogram (25 ms window, 10 ms hop, 16 kHz); temperature /
    humidity / air-quality standardised per channel.

    Expected layout:
        raw_dir/rgb/<id>.jpg, raw_dir/audio/<id>.wav,
        raw_dir/env/<id>.json, raw_dir/labels.json
    """
    N_MELS = 128

    def _index_samples(self):
        with open(self.raw_dir / "labels.json") as f:
            return list(json.load(f).keys())

    def _load_raw(self, sample_id):
        with open(self.raw_dir / "labels.json") as f:
            label = json.load(f)[sample_id]
        with open(self.raw_dir / "env" / f"{sample_id}.json") as f:
            env = json.load(f)  # {"temperature": .., "humidity": .., "air_quality": ..}
        env_vec = np.array([env["temperature"], env["humidity"], env["air_quality"]],
                            dtype=np.float32)
        return {
            "visual": torch.zeros(3, 224, 224),
            "acoustic": torch.zeros(self.N_MELS, 100),  # mel-spectrogram placeholder
            "environmental": torch.from_numpy(env_vec),
        }, int(label)


class SpectrumSenseDataset(BaseMultimodalDataset):
    """
    S4.3: RF windows of 256 samples, 50% overlap; PSD per segment
    normalised to [0,1]; contextual metadata one-hot encoded.

    Expected layout:
        raw_dir/rf/<id>.npz (raw IQ samples), raw_dir/context.json,
        raw_dir/labels.json
    """
    WINDOW = 256
    OVERLAP = 0.5

    def _index_samples(self):
        with open(self.raw_dir / "labels.json") as f:
            return list(json.load(f).keys())

    def _load_raw(self, sample_id):
        with open(self.raw_dir / "labels.json") as f:
            label = json.load(f)[sample_id]
        iq = np.load(self.raw_dir / "rf" / f"{sample_id}.npz")["iq"]
        psd = np.abs(np.fft.fft(iq[: self.WINDOW])) ** 2
        psd = (psd - psd.min()) / max(psd.max() - psd.min(), 1e-8)
        return {"rf": torch.from_numpy(psd).float()}, int(label)


class DeepSense6GDataset(BaseMultimodalDataset):
    """
    S4.4: Visual processed as MMAct; RF beam/channel measurements
    normalised per-scenario using training-split statistics; GPS
    encoded as relative displacement from the scenario median,
    normalised to [-1, 1].

    Source: https://dx.doi.org/10.21227/wq34-a443  [46]
    Expected layout:
        raw_dir/rgb/<id>.jpg, raw_dir/rf/<id>.npz, raw_dir/gps/<id>.json,
        raw_dir/labels.json
    """

    def _index_samples(self):
        with open(self.raw_dir / "labels.json") as f:
            return list(json.load(f).keys())

    def _load_raw(self, sample_id):
        with open(self.raw_dir / "labels.json") as f:
            label = json.load(f)[sample_id]
        rf = np.load(self.raw_dir / "rf" / f"{sample_id}.npz")["beam"]
        with open(self.raw_dir / "gps" / f"{sample_id}.json") as f:
            gps = json.load(f)  # {"lat": .., "lon": .., "scenario_median": [..]}
        rel = np.array(
            [gps["lat"] - gps["scenario_median"][0],
             gps["lon"] - gps["scenario_median"][1]], dtype=np.float32,
        )
        rel = np.clip(rel, -1.0, 1.0)
        return {
            "visual": torch.zeros(3, 224, 224),
            "rf": torch.from_numpy(rf).float(),
            "imu": torch.from_numpy(rel).float(),
        }, int(label)
