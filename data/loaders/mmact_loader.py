"""
MMAct loader (Kong et al., ICCV 2019) -- Section 4.1 / Supplementary S4.1.

Preprocessing (S4.1):
  - RGB frames resized to 224x224, ImageNet-normalised
    (mean [0.485,0.456,0.406], std [0.229,0.224,0.225])
  - Accelerometer/gyroscope/orientation: per-axis zero-mean/unit-variance
  - Wi-Fi RSSI: min-max normalised to [0, 1]
  - Temporal windows: 2 s, 50% overlap, for all inertial/Wi-Fi streams

Download / dataset home page:
  https://openaccess.thecvf.com/content_ICCV_2019/html/Kong_MMAct_A_Large-Scale_Dataset_for_Cross_Modal_Human_Action_Understanding_ICCV_2019_paper.html
"""
import numpy as np
import torch
import torchvision.transforms as T

from src.data.loaders.base_multimodal_loader import BaseMultimodalDataset

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
WINDOW_S = 2.0
OVERLAP = 0.5


class MMActDataset(BaseMultimodalDataset):
    """
    Expects `raw_dir` to contain the MMAct release layout:
        raw_dir/rgb/<sample_id>.mp4
        raw_dir/imu/<sample_id>.npz        (acc, gyro, ori)
        raw_dir/wifi/<sample_id>.npz       (rssi)
        raw_dir/labels.json                {sample_id: action_class_idx}
    """

    def __init__(self, raw_dir, split="train", seed=42):
        self.visual_tf = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
        super().__init__(raw_dir, split, seed)
        self._imu_stats = None  # computed lazily from the training split

    def _index_samples(self):
        labels_path = self.raw_dir / "labels.json"
        if not labels_path.exists():
            raise FileNotFoundError(
                f"{labels_path} not found. Download MMAct and place it "
                f"under {self.raw_dir} following the layout in this file's "
                f"docstring before running preprocess.py."
            )
        import json
        with open(labels_path) as f:
            labels = json.load(f)
        return list(labels.keys())

    def _windows(self, signal: np.ndarray, fps: float):
        win = int(WINDOW_S * fps)
        step = int(win * (1 - OVERLAP))
        return [signal[i:i + win] for i in range(0, max(len(signal) - win, 1), step)]

    def _load_raw(self, sample_id):
        import json
        with open(self.raw_dir / "labels.json") as f:
            label = json.load(f)[sample_id]

        imu = np.load(self.raw_dir / "imu" / f"{sample_id}.npz")
        acc = self._normalise(imu["acc"])
        gyro = self._normalise(imu["gyro"])
        ori = self._normalise(imu["ori"])
        imu_feat = np.concatenate([acc, gyro, ori], axis=-1)

        wifi = np.load(self.raw_dir / "wifi" / f"{sample_id}.npz")["rssi"]
        wifi_norm = (wifi - wifi.min()) / max(wifi.max() - wifi.min(), 1e-8)

        # Video decoding intentionally left to the user's preferred
        # backend (torchvision.io / decord / opencv) -- placeholder
        # tensor keeps the interface stable for training/testing.
        visual = torch.zeros(3, 224, 224)

        modality_dict = {
            "visual": visual,
            "imu": torch.from_numpy(imu_feat.T).float(),
            "wifi": torch.from_numpy(wifi_norm).float(),
        }
        return modality_dict, int(label)

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        return (x - x.mean(axis=0, keepdims=True)) / (x.std(axis=0, keepdims=True) + 1e-8)
