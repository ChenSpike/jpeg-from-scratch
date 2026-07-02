import numpy as np


def _generate_zigzag_indices(size: int = 8) -> np.ndarray:
    """
    Generates the (row, col) index order for a zigzag traversal of an
    NxN block, starting at (0,0) and alternating diagonal direction.

    This is a standard JPEG zigzag pattern: it walks anti-diagonals so
    that low-frequency DCT coefficients (top-left) come first and
    high-frequency ones (bottom-right, usually zero after quantization)
    come last — which is what makes the trailing run of zeros so long
    and RLE-friendly.
    """
    indices = [[] for _ in range(2 * size - 1)]
    for i in range(size):
        for j in range(size):
            diagonal = i + j
            if diagonal % 2 == 0:
                indices[diagonal].insert(0, (i, j))  # going up-right
            else:
                indices[diagonal].append((i, j))  # going down-left

    flat_order = [pos for diagonal in indices for pos in diagonal]
    return np.array(flat_order)


# Precomputed once at import time — the traversal order only depends on
# block size (always 8x8 here), not on any input data.
_ZIGZAG_INDICES = _generate_zigzag_indices(8)


def zigzag_scan(block_2d: np.ndarray) -> np.ndarray:
    """Flattens an 8x8 block into a length-64 1D array in zigzag order."""
    rows, cols = _ZIGZAG_INDICES[:, 0], _ZIGZAG_INDICES[:, 1]
    return block_2d[rows, cols]


def inverse_zigzag(flat: np.ndarray, size: int = 8) -> np.ndarray:
    """Reconstructs an 8x8 block from a length-64 1D zigzag-ordered array."""
    block_2d = np.zeros((size, size), dtype=flat.dtype)
    rows, cols = _ZIGZAG_INDICES[:, 0], _ZIGZAG_INDICES[:, 1]
    block_2d[rows, cols] = flat
    return block_2d
