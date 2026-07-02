import numpy as np
import rawpy


def load_raw_as_rgb(file_path: str) -> np.ndarray:
    """Loads and postprocesses a RAW image into an RGB array."""
    with rawpy.imread(file_path) as raw:
        return raw.postprocess(half_size=True, use_camera_wb=True)
