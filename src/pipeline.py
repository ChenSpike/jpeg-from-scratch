from dataclasses import dataclass

import numpy as np

from block_codec import QuantTable, decode_channel, encode_channel
from color import (
    chroma_downsample,
    chroma_upsample,
    merge_ycbcr_to_rgb,
    rgb_to_ycbcr,
)
from config import (
    CHROMA_SUBSAMPLE_MODE,
    JPEG_STD_CHROMA_QUANT_TABLE,
    JPEG_STD_LUMA_QUANT_TABLE,
)
from entropy_lite import decode_channel_symbols, encode_channel_symbols


@dataclass
class EncodedChannel:
    """Entropy-ready symbol representation of one channel (Y, Cb, or Cr)."""

    dc_diffs: list[float]
    ac_pairs_per_block: list[list[tuple[int, float]]]
    shape: tuple[int, int]  # channel shape *before* padding to 8x8 blocks
    sparsity: float  # % of zero coefficients after quantization


@dataclass
class EncodedImage:
    """Full entropy-ready representation of an encoded RGB image."""

    y: EncodedChannel
    cb: EncodedChannel
    cr: EncodedChannel
    original_shape: tuple[int, int]  # (h, w) of the source image
    quality_factor: int
    chroma_mode: str


class JpegPipeline:
    """
    Ties together color conversion, chroma subsampling, block-level
    DCT/quantization, zigzag scanning, and DC/AC symbol encoding into a
    single Y/Cb/Cr JPEG-style pipeline (Huffman coding not included).
    """

    def __init__(
        self, quality_factor: int = 50, chroma_mode: str = CHROMA_SUBSAMPLE_MODE
    ) -> None:
        self.quality_factor = quality_factor
        self.chroma_mode = chroma_mode
        self.luma_qtable = QuantTable(JPEG_STD_LUMA_QUANT_TABLE, quality_factor).scale()
        self.chroma_qtable = QuantTable(
            JPEG_STD_CHROMA_QUANT_TABLE, quality_factor
        ).scale()

    def _encode_single_channel(
        self, channel: np.ndarray, q_table: np.ndarray
    ) -> EncodedChannel:
        """Runs one channel through block codec + entropy stages."""
        blocks, sparsity = encode_channel(channel, q_table)
        dc_diffs, ac_pairs_per_block = encode_channel_symbols(blocks)
        return EncodedChannel(
            dc_diffs=dc_diffs,
            ac_pairs_per_block=ac_pairs_per_block,
            shape=channel.shape,
            sparsity=sparsity,
        )

    def _decode_single_channel(
        self, encoded: EncodedChannel, q_table: np.ndarray
    ) -> np.ndarray:
        """Inverts _encode_single_channel back to pixel values."""
        blocks = decode_channel_symbols(encoded.dc_diffs, encoded.ac_pairs_per_block)
        return decode_channel(blocks, q_table, encoded.shape)

    def encode_image(self, rgb_image: np.ndarray) -> EncodedImage:
        """Encodes an RGB image into Y/Cb/Cr entropy-ready symbols."""
        y, cb, cr = rgb_to_ycbcr(rgb_image)

        cb_sub = chroma_downsample(cb, mode=self.chroma_mode)
        cr_sub = chroma_downsample(cr, mode=self.chroma_mode)

        return EncodedImage(
            y=self._encode_single_channel(y, self.luma_qtable),
            cb=self._encode_single_channel(cb_sub, self.chroma_qtable),
            cr=self._encode_single_channel(cr_sub, self.chroma_qtable),
            original_shape=rgb_image.shape[:2],
            quality_factor=self.quality_factor,
            chroma_mode=self.chroma_mode,
        )

    def decode_image(self, encoded: EncodedImage) -> np.ndarray:
        """Decodes Y/Cb/Cr entropy-ready symbols back into an RGB image."""
        y = self._decode_single_channel(encoded.y, self.luma_qtable)
        cb_sub = self._decode_single_channel(encoded.cb, self.chroma_qtable)
        cr_sub = self._decode_single_channel(encoded.cr, self.chroma_qtable)

        cb = chroma_upsample(cb_sub, encoded.original_shape, mode=encoded.chroma_mode)
        cr = chroma_upsample(cr_sub, encoded.original_shape, mode=encoded.chroma_mode)

        return merge_ycbcr_to_rgb(y, cb.astype(np.uint8), cr.astype(np.uint8))

    def overall_sparsity(self, encoded: EncodedImage) -> float:
        """Weighted-average sparsity across Y/Cb/Cr, for reporting."""
        channels = [encoded.y, encoded.cb, encoded.cr]
        total_coefs = sum(len(c.ac_pairs_per_block) * 64 for c in channels)
        # sparsity is stored as a %, recover approximate zero-count and re-average
        zero_coefs = sum(
            c.sparsity / 100 * len(c.ac_pairs_per_block) * 64 for c in channels
        )
        return (zero_coefs / total_coefs) * 100 if total_coefs else 0.0
