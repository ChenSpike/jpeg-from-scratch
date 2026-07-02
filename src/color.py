import cv2
import numpy as np

from config import CHROMA_SUBSAMPLE_MODE


def rgb_to_ycbcr(rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts RGB to (Y, Cb, Cr) channels."""
    y_cr_cb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    return y_cr_cb[:, :, 0], y_cr_cb[:, :, 2], y_cr_cb[:, :, 1]


def merge_ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Merges (Y, Cb, Cr) channels of matching shape back into RGB."""
    y_cr_cb = cv2.merge([y, cr, cb])
    return cv2.cvtColor(y_cr_cb, cv2.COLOR_YCrCb2RGB)


def _subsample_shape(shape: tuple[int, int], mode: str) -> tuple[int, int]:
    """Computes the target (h, w) for a given subsampling mode."""
    h, w = shape
    if mode == "4:2:0":
        return (max(1, h // 2), max(1, w // 2))
    if mode == "4:2:2":
        return (h, max(1, w // 2))
    if mode == "4:4:4":
        return (h, w)
    raise ValueError(f"Unsupported chroma subsampling mode: {mode}")


def chroma_downsample(
    channel: np.ndarray, mode: str = CHROMA_SUBSAMPLE_MODE
) -> np.ndarray:
    """
    Downsamples a single chroma channel (Cb or Cr) according to `mode`.

    Uses area-averaging interpolation, matching how real JPEG encoders
    typically average neighboring samples rather than dropping them.
    """
    target_h, target_w = _subsample_shape(channel.shape, mode)
    return cv2.resize(
        channel, (target_w, target_h), interpolation=cv2.INTER_AREA
    )


def chroma_upsample(
    channel: np.ndarray,
    target_shape: tuple[int, int],
    mode: str = CHROMA_SUBSAMPLE_MODE,
) -> np.ndarray:
    """
    Upsamples a subsampled chroma channel back to `target_shape` (h, w).

    `mode` is accepted for symmetry with chroma_downsample and future
    mode-specific upsampling strategies, but linear interpolation works
    for reconstructing any of the supported modes.
    """
    target_h, target_w = target_shape
    return cv2.resize(
        channel, (target_w, target_h), interpolation=cv2.INTER_LINEAR
    )
