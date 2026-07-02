from typing import Iterator

import cv2
import numpy as np


class QuantTable:
    """Generates a scaled quantization table for a given quality factor."""

    def __init__(self, base_table: np.ndarray, quality_factor: int = 50) -> None:
        self.base_table = base_table
        self.quality_factor = max(1, min(100, quality_factor))

    def scale(self) -> np.ndarray:
        """Scales the base quantization table by the quality factor."""
        if self.quality_factor < 50:
            scale = 5000 / self.quality_factor
        else:
            scale = 200 - self.quality_factor * 2

        q_table = np.floor((self.base_table * scale + 50) / 100)
        q_table[q_table == 0] = 1  # Avoid division by zero
        return q_table.astype(np.float32)


def pad_channel(channel: np.ndarray) -> np.ndarray:
    """Pads the boundary of a single channel to be a multiple of 8."""
    h, w = channel.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    return np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")


def unpad_channel(channel: np.ndarray, original_shape: tuple[int, int]) -> np.ndarray:
    """Crops a padded channel back down to its original shape."""
    return channel[: original_shape[0], : original_shape[1]]


def iter_blocks(channel: np.ndarray) -> Iterator[tuple[int, int, np.ndarray]]:
    """
    Iterates over a padded channel in 8x8 blocks.

    Yields (row_offset, col_offset, block) so callers can reassemble
    blocks back into their original position without recomputing indices.
    """
    h, w = channel.shape
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            yield i, j, channel[i : i + 8, j : j + 8]


def encode_block(block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """Transforms and quantizes a single 8x8 block using the given q_table."""
    # Level shift
    shifted = block.astype(np.float32) - 128.0
    # 2D-DCT
    dct_coeffs = cv2.dct(shifted)
    # Quantization
    return np.round(dct_coeffs / q_table)


def decode_block(quantized_block: np.ndarray, q_table: np.ndarray) -> np.ndarray:
    """Inverse quantization, IDCT, and level restore for a single 8x8 block."""
    dct_coeffs = quantized_block * q_table
    shifted_block = cv2.idct(dct_coeffs)
    block = np.clip(shifted_block + 128.0, 0, 255)
    return block.astype(np.uint8)


def encode_channel(
    channel: np.ndarray, q_table: np.ndarray
) -> tuple[list[np.ndarray], float]:
    """
    Encodes a channel block-by-block and counts zero coefficients.

    Works for Y, Cb, or Cr — pass the appropriate q_table for the channel.

    Returns:
        tuple: (list of compressed 8x8 blocks, sparsity_ratio)
    """
    padded = pad_channel(channel)
    compressed_blocks = []
    total_coefs = 0
    zero_coefs = 0

    for _, _, block in iter_blocks(padded):
        quantized = encode_block(block, q_table)
        compressed_blocks.append(quantized)
        total_coefs += 64
        zero_coefs += np.sum(quantized == 0)

    sparsity_ratio = (zero_coefs / total_coefs) * 100
    return compressed_blocks, sparsity_ratio


def decode_channel(
    compressed_blocks: list[np.ndarray],
    q_table: np.ndarray,
    original_shape: tuple[int, int],
) -> np.ndarray:
    """Reconstructs a channel from 8x8 blocks and removes padding."""
    padded_h = (original_shape[0] + 7) // 8 * 8
    padded_w = (original_shape[1] + 7) // 8 * 8
    reconstructed = np.zeros((padded_h, padded_w), dtype=np.uint8)

    block_idx = 0
    for i in range(0, padded_h, 8):
        for j in range(0, padded_w, 8):
            reconstructed[i : i + 8, j : j + 8] = decode_block(
                compressed_blocks[block_idx], q_table
            )
            block_idx += 1

    return unpad_channel(reconstructed, original_shape)
