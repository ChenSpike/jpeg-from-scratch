import cv2
import numpy as np
import rawpy
from config import JPEG_STD_LUMA_QUANT_TABLE


class JpegEncoder:

    def __init__(
            self, 
            quality_factor: int = 50, 
            base_q_table: np.ndarray = None
        ) -> None:
            self.quality_factor = max(1, min(100, quality_factor))

            self.base_table = (
                base_q_table
                if base_q_table is not None
                else JPEG_STD_LUMA_QUANT_TABLE
            )
            self.q_table = self._generate_quantization_table()

    def _generate_quantization_table(self) -> np.ndarray:
        """Scales base quantization table by quality factor."""
        if self.quality_factor < 50:
            scale = 5000 / self.quality_factor
        else:
            scale = 200 - self.quality_factor * 2

        q_table = np.floor((self.base_table * scale + 50) / 100)
        q_table[q_table == 0] = 1  # Avoid division by zero
        return q_table.astype(np.float32)

    @staticmethod
    def pad_channel(channel: np.ndarray) -> np.ndarray:
        """Pads the boundary of a single channel to be a multiple of 8."""
        h, w = channel.shape
        pad_h = (8 - (h % 8)) % 8
        pad_w = (8 - (w % 8)) % 8
        return np.pad(channel, ((0, pad_h), (0, pad_w)), mode="edge")

    def encode_block(self, block: np.ndarray) -> np.ndarray:
        """Transforms and quantizes a single 8x8 block."""
        # Level shift
        shifted = block.astype(np.float32) - 128.0
        # 2D-DCT
        dct_coeffs = cv2.dct(shifted)
        # Quantization
        return np.round(dct_coeffs / self.q_table)

    def encode_channel(self, channel: np.ndarray) -> tuple[list[np.ndarray], float]:
        """
        Encodes channel block-by-block and counts zero coefficients.

        Returns:
            tuple: (list of compressed 8x8 blocks, sparsity_ratio)
        """
        padded = self.pad_channel(channel)
        h, w = padded.shape
        compressed_blocks = []
        total_coefs = 0
        zero_coefs = 0

        # Block processing pipeline
        for i in range(0, h, 8):
            for j in range(0, w, 8):
                block = padded[i : i + 8, j : j + 8]
                quantized = self.encode_block(block)

                compressed_blocks.append(quantized)
                total_coefs += 64
                zero_coefs += np.sum(quantized == 0)

        sparsity_ratio = (zero_coefs / total_coefs) * 100
        return compressed_blocks, sparsity_ratio
    
    def decode_block(self, quantized_block: np.ndarray) -> np.ndarray:
        """Inverse quantization, IDCT, and level restore."""
        dct_coeffs = quantized_block * self.q_table
        shifted_block = cv2.idct(dct_coeffs)
        block = np.clip(shifted_block + 128.0, 0, 255)
        return block.astype(np.uint8)

    def decode_channel(
        self, 
        compressed_blocks: list[np.ndarray], 
        original_shape: tuple[int, int]
    ) -> np.ndarray:
        """Reconstructs channel from 8x8 blocks and removes padding."""
        padded_h = (original_shape[0] + 7) // 8 * 8
        padded_w = (original_shape[1] + 7) // 8 * 8
        reconstructed = np.zeros((padded_h, padded_w), dtype=np.uint8)

        block_idx = 0
        for i in range(0, padded_h, 8):
            for j in range(0, padded_w, 8):
                reconstructed[i : i + 8, j : j + 8] = self.decode_block(
                    compressed_blocks[block_idx]
                )
                block_idx += 1

        return reconstructed[: original_shape[0], : original_shape[1]]


def load_raw_as_rgb(file_path: str) -> np.ndarray:
    """Loads and postprocesses a RAW image."""
    with rawpy.imread(file_path) as raw:
        return raw.postprocess(half_size=True, use_camera_wb=True)
    

def rgb_to_ycbcr(rgb_image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts RGB to YCbCr channels."""
    y_cr_cb = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
    return y_cr_cb[:, :, 0], y_cr_cb[:, :, 2], y_cr_cb[:, :, 1]


def merge_ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    """Merges YCbCr channels back to RGB."""
    y_cr_cb = cv2.merge([y, cr, cb])
    return cv2.cvtColor(y_cr_cb, cv2.COLOR_YCrCb2RGB)