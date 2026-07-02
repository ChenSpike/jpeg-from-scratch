import numpy as np

from scan import zigzag_scan, inverse_zigzag


def dc_differential_encode(dc_values: list[float]) -> list[float]:
    """
    Encodes a list of per-block DC coefficients as differences from the
    previous block's DC value. The first value is stored as-is.

    Adjacent 8x8 blocks tend to have similar average brightness, so the
    differences cluster near zero and compress better than raw DC values.
    """
    if not dc_values:
        return []
    diffs = [dc_values[0]]
    for prev, curr in zip(dc_values, dc_values[1:]):
        diffs.append(curr - prev)
    return diffs


def dc_differential_decode(diffs: list[float]) -> list[float]:
    """Reconstructs per-block DC coefficients from their differences."""
    if not diffs:
        return []
    dc_values = [diffs[0]]
    for diff in diffs[1:]:
        dc_values.append(dc_values[-1] + diff)
    return dc_values


def rle_encode(ac_values: np.ndarray) -> list[tuple[int, float]]:
    """
    Run-length encodes a 1D array of AC coefficients (zigzag order) as
    (zero_run_length, value) pairs, one per non-zero coefficient.

    Trailing zeros after the last non-zero value are not stored at all —
    this is equivalent to JPEG's End-Of-Block marker: rle_decode simply
    leaves the rest of the block as zero.
    """
    pairs = []
    zero_run = 0
    for v in ac_values:
        if v == 0:
            zero_run += 1
        else:
            pairs.append((zero_run, float(v)))
            zero_run = 0
    return pairs


def rle_decode(pairs: list[tuple[int, float]], length: int = 63) -> np.ndarray:
    """
    Reconstructs a length-`length` 1D array of AC coefficients from
    (zero_run_length, value) pairs. Any position past the last pair
    (i.e. the implicit End-Of-Block) is left as zero.
    """
    values = np.zeros(length, dtype=np.float32)
    idx = 0
    for zero_run, value in pairs:
        idx += zero_run
        values[idx] = value
        idx += 1
    return values


def encode_block_symbols(
    quantized_block: np.ndarray,
) -> tuple[float, list[tuple[int, float]]]:
    """
    Converts a single quantized 8x8 block into (dc_value, ac_pairs) —
    the pre-Huffman symbol representation of one block.
    """
    flat = zigzag_scan(quantized_block)
    dc_value = float(flat[0])
    ac_pairs = rle_encode(flat[1:])
    return dc_value, ac_pairs


def decode_block_symbols(
    dc_value: float, ac_pairs: list[tuple[int, float]]
) -> np.ndarray:
    """Reconstructs a single quantized 8x8 block from (dc_value, ac_pairs)."""
    ac_values = rle_decode(ac_pairs, length=63)
    flat = np.concatenate(([dc_value], ac_values)).astype(np.float32)
    return inverse_zigzag(flat)


def encode_channel_symbols(
    compressed_blocks: list[np.ndarray],
) -> tuple[list[float], list[list[tuple[int, float]]]]:
    """
    Converts all quantized blocks of a channel into the entropy-ready
    symbol stream: DC values (differentially encoded) and one AC
    run-length pair list per block.
    """
    dc_values = []
    ac_pairs_per_block = []
    for block in compressed_blocks:
        dc_value, ac_pairs = encode_block_symbols(block)
        dc_values.append(dc_value)
        ac_pairs_per_block.append(ac_pairs)

    dc_diffs = dc_differential_encode(dc_values)
    return dc_diffs, ac_pairs_per_block


def decode_channel_symbols(
    dc_diffs: list[float],
    ac_pairs_per_block: list[list[tuple[int, float]]],
) -> list[np.ndarray]:
    """Reconstructs all quantized 8x8 blocks of a channel from symbols."""
    dc_values = dc_differential_decode(dc_diffs)
    return [
        decode_block_symbols(dc, ac_pairs)
        for dc, ac_pairs in zip(dc_values, ac_pairs_per_block)
    ]
