import cv2
import numpy as np


def save_image(rgb_array: np.ndarray, output_path: str) -> None:
    """
    Saves an RGB array as an image file (format inferred from the
    extension in output_path, e.g. .png or .jpg).

    Note: this writes the reconstructed pixels for visual inspection
    only. Since this pipeline does not implement entropy (Huffman)
    coding, the resulting file size does NOT reflect a real compressed
    JPEG's size — use rd_curve.py / metrics.py for compression-quality
    analysis instead.
    """
    bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
    success = cv2.imwrite(output_path, bgr_array)
    if not success:
        raise IOError(f"Failed to save image to {output_path}")
