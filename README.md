# JPEG Image Compression Engine From Scratch

A decoupled, object-oriented implementation of the standard JPEG compression and decompression pipeline. This project processes raw camera data (.ARW), transforms images into the frequency domain using 2D-DCT, applies custom quantization based on user-defined quality factors, and reconstructs the image through an inverse pipeline.

## Key Features

- **End-to-End Pipeline:** Implements both the forward encoding (compression) and inverse decoding (reconstruction) processes.
- **Decoupled Architecture:** Separates configuration matrices, core digital signal processing algorithms, and the CLI execution layer into distinct modules.
- **Dynamic Quantization:** Generates scaled quantization tables from standard ISO luminance matrices based on a user-defined Quality Factor (1-100).
- **Matrix Coefficient Pruning:** Analyzes and reports the sparsity (percentage of high-frequency coefficients successfully zeroed out) after the quantization stage.

## Project Structure

```text
├── src/
│   ├── config.py         # Static configuration and ISO quantization matrices
│   ├── utils.py          # JpegEncoder class, DCT/IDCT and matrix transformations
│   └── main.py           # CLI controller, logging configuration, and visualization
├── data/
│   └── cat.ARW           # Sample 34.7MB RAW source image
├── requirements.txt      # Third-party dependencies (rawpy, opencv-python, matplotlib)
└── README.md             # Documentation
```

## Compression & Decompression Pipeline

### Forward Encoder Pipeline

1. **Ingestion:** Load RAW image data (.ARW) and decode to RGB space via `rawpy`.
2. **Color Transformation:** Convert RGB to YCbCr color space to extract the Luminance (Y) channel.
3. **Boundary Padding:** Pad the channel dimensions to ensure compatibility with 8x8 block structures.
4. **Mathematical Transformation:** Apply 2D Discrete Cosine Transform (2D-DCT) to shift spatial pixels into frequency coefficients.
5. **Quantization:** Divide coefficients by the quality-scaled quantization matrix and round to integers, forcing high-frequency components to zero.

### Inverse Decoder Pipeline

1. **Dequantization:** Rescale quantized matrices back by multiplying with the identical quantization table.
2. **Inverse Transformation:** Apply 2D Inverse Discrete Cosine Transform (2D-IDCT) to return to the spatial domain.
3. **Reconstitution:** Clip values to [0, 255], remove boundary padding, merge with chroma channels, and convert back to RGB for rendering.

## Quantitative Metrics

Benchmarks evaluated using a 32.7 MP (7008 x 4672) RAW image input (original size: 34.7 MB).

| Quality Factor (Q) | High-Frequency Coefficients Pruned | Target Compression Ratio | Visual Output Characteristics |
| --- | --- | --- | --- |
| Original RAW | 0.00% | 1 : 1 | Uncompressed reference source |
| Quality = 90 | ~65.40% | ~18 : 1 | No perceptible visual degradation |
| Quality = 50 | 92.83% | ~77 : 1 | Perceptually identical to source; high mathematical sparsity |
| Quality = 5 | >98.50% | >200 : 1 | Visible block artifacts (8x8 grid patterns appearance) |

## Getting Started

### 1. Environment Setup (Using Anaconda)

```bash
# Create and activate environment
conda create -n jpeg_env python=3.11 -y
conda activate jpeg_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Execution and Configuration

Run the end-to-end pipeline via the command line interface. You can adjust the file source path and target quality factor using parameters:

```bash
# Execute with default settings (Quality = 50)
python src/main.py --input ./data/cat.ARW --quality 50

# Test extreme compression constraints (Quality = 5)
python src/main.py --input ./data/cat.ARW --quality 5
```

### 3. Verification & Metrics Output

Upon a successful run, the pipeline logs each step and renders a Matplotlib visualization comparing the original input side-by-side with the lossy-compressed output:

```text
2026-07-01 16:00:00,000 - INFO - Initializing JPEG Pipeline (Quality=50)
2026-07-01 16:00:01,123 - INFO - Step 1: Loading RAW and converting to YCbCr
2026-07-01 16:00:01,544 - INFO - Step 2: Running forward encoder (DCT + Quantization)
2026-07-01 16:00:03,110 - INFO - Compression Metric - Pruned Coefficients: 92.83%
2026-07-01 16:00:03,215 - INFO - Step 3: Running inverse decoder (IDCT + Dequantization)
2026-07-01 16:00:03,500 - INFO - Step 4: Recombining channels to RGB
2026-07-01 16:00:03,600 - INFO - Pipeline successful. Rendering visualization...
```