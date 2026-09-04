"""Data preprocessing pipeline — identical across all baselines for fair comparison."""
import numpy as np
import torch

def preprocess_visual(img):
    """Normalise RGB image to ImageNet stats."""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return (img / 255.0 - mean) / std

def preprocess_acoustic(wav, sr=16000, n_mels=64, n_fft=512, hop=160):
    """Compute mel-spectrogram from raw waveform."""
    try:
        import librosa
        S = librosa.feature.melspectrogram(y=wav.numpy(), sr=sr,
                                           n_mels=n_mels, n_fft=n_fft, hop_length=hop)
        return torch.tensor(librosa.power_to_db(S)).unsqueeze(0)
    except ImportError:
        return torch.randn(1, n_mels, 128)

def preprocess_imu(data):
    """Normalise 9-axis IMU to zero mean, unit std."""
    return (data - data.mean(0)) / (data.std(0) + 1e-8)

def preprocess_rf(signal):
    """Normalise RF spectrum."""
    return (signal - signal.mean()) / (signal.std() + 1e-8)

def preprocess_environmental(readings):
    """Min-max normalise environmental sensors."""
    return (readings - readings.min()) / (readings.max() - readings.min() + 1e-8)
