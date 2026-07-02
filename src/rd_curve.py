import matplotlib.pyplot as plt
import numpy as np

from config import CHROMA_SUBSAMPLE_MODE
from metrics import SUPPORTED_METRICS, compute_metric
from pipeline import JpegPipeline

_METRIC_LABELS = {"psnr": "PSNR (dB)", "ssim": "SSIM"}


def sweep_quality(
    rgb_image: np.ndarray,
    qualities: list[int],
    metric: str = "psnr",
    chroma_mode: str = CHROMA_SUBSAMPLE_MODE,
) -> tuple[list[int], list[float]]:
    """
    Runs the full encode/decode pipeline once per quality factor in
    `qualities` and scores each reconstruction against the original
    using `metric` ("psnr" or "ssim").

    Returns (qualities, scores), aligned by index.
    """
    metric = metric.lower()
    if metric not in SUPPORTED_METRICS:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose from {SUPPORTED_METRICS}."
        )

    scores = []
    for q in qualities:
        pipeline = JpegPipeline(quality_factor=q, chroma_mode=chroma_mode)
        encoded = pipeline.encode_image(rgb_image)
        reconstructed = pipeline.decode_image(encoded)
        scores.append(compute_metric(rgb_image, reconstructed, metric=metric))

    return list(qualities), scores


def plot_rd_curve(
    qualities: list[int],
    scores: list[float],
    metric: str = "psnr",
    output_path: str = None,
) -> plt.Figure:
    """
    Plots metric score vs. quality factor.

    Note: the x-axis is the JPEG quality factor, not a measured bpp —
    this pipeline has no entropy (Huffman) stage, so a true rate-distortion
    curve (bpp on the x-axis) isn't available.
    """
    metric = metric.lower()
    ylabel = _METRIC_LABELS.get(metric, metric.upper())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(qualities, scores, marker="o")
    ax.set_xlabel("Quality Factor")
    ax.set_ylabel(ylabel)
    ax.set_title(
        f"{ylabel} vs. Quality Factor"
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig


def plot_metric_curve(
    qualities: list[int],
    scores_by_metric: dict[str, list[float]],
    output_path: str = None,
) -> plt.Figure:
    """
    Plots one or two metrics vs. quality factor on the same figure.

    scores_by_metric: e.g. {"psnr": [...]} for a single metric, or
    {"psnr": [...], "ssim": [...]} to overlay both on twin y-axes
    (PSNR in dB and SSIM in [0,1] don't share a scale).
    """
    metrics_present = list(scores_by_metric.keys())
    for m in metrics_present:
        if m not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{m}'. Choose from {SUPPORTED_METRICS}."
            )

    fig, ax1 = plt.subplots(figsize=(8, 6))
    ax1.set_xlabel("Quality Factor")

    if "psnr" in metrics_present:
        color1 = "tab:blue"
        ax1.plot(
            qualities, scores_by_metric["psnr"], marker="o", color=color1, label="PSNR"
        )
        ax1.set_ylabel(_METRIC_LABELS["psnr"], color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        if "ssim" in metrics_present:
            ax2 = ax1.twinx()
            color2 = "tab:red"
            ax2.plot(
                qualities,
                scores_by_metric["ssim"],
                marker="s",
                color=color2,
                label="SSIM",
            )
            ax2.set_ylabel(_METRIC_LABELS["ssim"], color=color2)
            ax2.tick_params(axis="y", labelcolor=color2)
    elif "ssim" in metrics_present:
        color1 = "tab:blue"
        ax1.plot(
            qualities, scores_by_metric["ssim"], marker="o", color=color1, label="SSIM"
        )
        ax1.set_ylabel(_METRIC_LABELS["ssim"], color=color1)

    title_metrics = " & ".join(m.upper() for m in metrics_present)
    ax1.set_title(
        f"{title_metrics} vs. Quality Factor"
    )
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=150, bbox_inches="tight")

    return fig