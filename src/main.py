import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt

from metrics import compute_metric
from output_io import save_image
from pipeline import JpegPipeline
from raw_io import load_raw_as_rgb
from rd_curve import plot_metric_curve

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    parser = argparse.ArgumentParser(description="JPEG End-to-End Pipeline")
    parser.add_argument(
        "--input", "-i", type=str, required=True, help="Path to the input RAW image (e.g., .ARW file)"
    )
    parser.add_argument(
        "--quality",
        "-q",
        type=int,
        nargs="+",
        default=[50],
        help="One or more quality factors (0-100), default: 50",
    )
    parser.add_argument(
        "--subsampling",
        type=str,
        default="4:2:0",
        choices=["4:2:0", "4:2:2", "4:4:4"],
        help="Chroma subsampling mode, default: 4:2:0",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save reconstructed image(s) (and curve plot, if --plot-curve is set) to --output-dir",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display reconstructed image(s) (and curve plot, if --plot-curve is set) in a window",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Directory for --save output, default: outputs",
    )
    parser.add_argument(
        "--plot-curve",
        nargs="?",
        const="psnr",
        default=None,
        choices=["psnr", "ssim", "both"],
        help="Plot a quality-vs-metric curve. Bare flag defaults to psnr. "
        "Requires --save and/or --show to produce output.",
    )
    return parser.parse_args()


def validate_args(args) -> None:
    """Validates CLI arguments, exiting with a clear message on failure."""
    if not os.path.isfile(args.input):
        logging.error(f"Input file not found: {args.input}")
        sys.exit(1)

    for q in args.quality:
        if not (0 <= q <= 100):
            logging.error(f"Quality factor must be between 0 and 100, got: {q}")
            sys.exit(1)
        if q == 0:
            logging.warning(
                "Quality factor 0 is internally clamped to 1 (minimum valid quantization scale)."
            )

    if args.plot_curve and len(args.quality) < 2:
        logging.warning(
            "Only one --quality value given; the curve will show a single point. "
            "Pass multiple --quality values for a meaningful curve."
        )

    if args.plot_curve and not (args.save or args.show):
        logging.warning(
            "--plot-curve was set but neither --save nor --show was given; "
            "the curve will be computed but not saved or displayed."
        )


def run_quality_sweep(rgb_image, qualities: list[int], subsampling: str) -> list[dict]:
    """
    Encodes and decodes rgb_image once per quality factor. Returns a list
    of per-quality results (reconstructed image + sparsity + metrics),
    computed once and reused for saving, display, and curve plotting.
    """
    results = []
    for q in qualities:
        pipeline = JpegPipeline(quality_factor=q, chroma_mode=subsampling)
        encoded = pipeline.encode_image(rgb_image)
        decoded = pipeline.decode_image(encoded)
        results.append(
            {
                "quality": q,
                "image": decoded,
                "sparsity": pipeline.overall_sparsity(encoded),
                "psnr": compute_metric(rgb_image, decoded, "psnr"),
                "ssim": compute_metric(rgb_image, decoded, "ssim"),
            }
        )
        logging.info(
            f"Quality={q}: sparsity={results[-1]['sparsity']:.2f}%  "
            f"PSNR={results[-1]['psnr']:.2f}dB  SSIM={results[-1]['ssim']:.4f}"
        )
    return results


def save_results(results: list[dict], input_path: str, subsampling: str, output_dir: str) -> None:
    """Saves one reconstructed image per quality factor."""
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(input_path).stem
    sub_tag = subsampling.replace(":", "")
    for r in results:
        filename = f"{stem}_q{r['quality']}_{sub_tag}.png"
        out_path = os.path.join(output_dir, filename)
        save_image(r["image"], out_path)
        logging.info(f"Saved reconstructed image to {out_path}")


def build_comparison_figure(original_rgb, results: list[dict], subsampling: str) -> plt.Figure:
    """Builds a 1xN panel: original followed by one reconstruction per quality."""
    n_panels = len(results) + 1
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Original", fontsize=12)
    axes[0].axis("off")

    for ax, r in zip(axes[1:], results):
        ax.imshow(r["image"])
        ax.set_title(
            f"Q={r['quality']} ({subsampling})\n"
            f"PSNR={r['psnr']:.1f}dB  SSIM={r['ssim']:.3f}",
            fontsize=10,
        )
        ax.axis("off")

    fig.tight_layout()
    return fig


def build_curve_figure(results: list[dict], metric_choice: str) -> plt.Figure:
    """Builds the quality-vs-metric curve figure for the requested metric(s)."""
    metrics_needed = ["psnr", "ssim"] if metric_choice == "both" else [metric_choice]
    qualities = [r["quality"] for r in results]
    scores_by_metric = {m: [r[m] for r in results] for m in metrics_needed}
    return plot_metric_curve(qualities, scores_by_metric)


def main():
    args = parse_args()
    validate_args(args)

    try:
        logging.info(f"Loading RAW image: {args.input}")
        rgb_img = load_raw_as_rgb(args.input)

        logging.info(
            f"Running pipeline (qualities={args.quality}, subsampling={args.subsampling})"
        )
        results = run_quality_sweep(rgb_img, args.quality, args.subsampling)

        if args.save:
            save_results(results, args.input, args.subsampling, args.output_dir)

        figures_to_show = []

        if args.show:
            figures_to_show.append(build_comparison_figure(rgb_img, results, args.subsampling))

        if args.plot_curve:
            curve_fig = build_curve_figure(results, args.plot_curve)

            if args.save:
                os.makedirs(args.output_dir, exist_ok=True)
                stem = Path(args.input).stem
                curve_path = os.path.join(
                    args.output_dir, f"{stem}_curve_{args.plot_curve}.png"
                )
                curve_fig.savefig(curve_path, dpi=150, bbox_inches="tight")
                logging.info(f"Saved curve plot to {curve_path}")

            if args.show:
                figures_to_show.append(curve_fig)

        if figures_to_show:
            plt.show()

        logging.info("Pipeline complete.")

    except Exception as e:
        logging.error(f"Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()