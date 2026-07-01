import argparse
import logging
import sys
import matplotlib.pyplot as plt
from utils import JpegEncoder, load_raw_as_rgb, merge_ycbcr_to_rgb, rgb_to_ycbcr

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def display_results(original_rgb, compressed_rgb, quality, sparsity):
    """Plots original vs compressed results."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original Image (RAW Source)", fontsize=14)
    axes[0].axis("off")

    axes[1].imshow(compressed_rgb)
    axes[1].set_title(
        f"Compressed Output (Quality = {quality})\nPruned Coefficients: {sparsity:.2f}%",
        fontsize=14,
    )
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="JPEG End-to-End Pipeline")
    parser.add_argument(
        "--input", type=str, required=True, help="Path to the input RAW image (e.g., .ARW file)"
    )
    parser.add_argument(
        "--quality", type=int, default=50, help="Quality factor (1-100), default is 50"
    )
    args = parser.parse_args()

    try:
        logging.info(f"Initializing JPEG Pipeline (Quality={args.quality})")
        encoder = JpegEncoder(quality_factor=args.quality)

        logging.info("Step 1: Loading RAW and converting to YCbCr")
        rgb_img = load_raw_as_rgb(args.input)
        y, cb, cr = rgb_to_ycbcr(rgb_img)

        logging.info("Step 2: Running forward encoder (DCT + Quantization)")
        compressed_blocks, sparsity = encoder.encode_channel(y)
        logging.info(f"Compression Metric - Pruned Coefficients: {sparsity:.2f}%")

        logging.info("Step 3: Running inverse decoder (IDCT + Dequantization)")
        y_reconstructed = encoder.decode_channel(compressed_blocks, y.shape)

        logging.info("Step 4: Recombining channels to RGB")
        compressed_rgb = merge_ycbcr_to_rgb(y_reconstructed, cb, cr)

        logging.info("Pipeline successful. Rendering visualization...")
        display_results(rgb_img, compressed_rgb, args.quality, sparsity)

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()