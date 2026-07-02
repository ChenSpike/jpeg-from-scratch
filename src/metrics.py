import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

SUPPORTED_METRICS = ("psnr", "ssim")


def compute_psnr(
    original: np.ndarray, reconstructed: np.ndarray, data_range: int = 255
) -> float:
    """Peak Signal-to-Noise Ratio in dB. Higher is better (less distortion)."""
    return float(
        peak_signal_noise_ratio(original, reconstructed, data_range=data_range)
    )


def compute_ssim(
    original: np.ndarray, reconstructed: np.ndarray, data_range: int = 255
) -> float:
    """Structural Similarity Index, in [-1, 1]. Higher is better."""
    return float(
        structural_similarity(
            original, reconstructed, channel_axis=-1, data_range=data_range
        )
    )


def compute_metric(
    original: np.ndarray,
    reconstructed: np.ndarray,
    metric: str = "psnr",
    data_range: int = 255,
) -> float:
    """
    Computes an image-quality metric between an original and reconstructed
    RGB image. `metric` is one of SUPPORTED_METRICS ("psnr" or "ssim").
    """
    metric = metric.lower()
    if metric == "psnr":
        return compute_psnr(original, reconstructed, data_range)
    if metric == "ssim":
        return compute_ssim(original, reconstructed, data_range)
    raise ValueError(
        f"Unsupported metric '{metric}'. Choose from {SUPPORTED_METRICS}."
    )
