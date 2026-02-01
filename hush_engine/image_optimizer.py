"""
Image Optimizer - Lossless compression using Zopfli (PNG) and MozJPEG (JPEG).

All libraries have commercial-friendly licenses:
- zopfli: Apache 2.0 (Google)
- mozjpeg-lossless-optimization: BSD-3

Usage:
    from image_optimizer import optimize_image
    optimize_image("/path/to/image.png")  # Optimizes in-place
    optimize_image("/path/to/image.jpg", "/path/to/output.jpg")  # To different file
"""

import sys
from pathlib import Path
from typing import Optional

# Optional imports - optimization is best-effort
try:
    import zopfli
    ZOPFLI_AVAILABLE = True
except ImportError:
    ZOPFLI_AVAILABLE = False
    sys.stderr.write("[ImageOptimizer] zopfli not installed, PNG optimization disabled\n")

try:
    from mozjpeg_lossless_optimization import optimize as mozjpeg_optimize
    MOZJPEG_AVAILABLE = True
except ImportError:
    MOZJPEG_AVAILABLE = False
    sys.stderr.write("[ImageOptimizer] mozjpeg-lossless-optimization not installed, JPEG optimization disabled\n")


def optimize_png(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Optimize PNG using Zopfli compression.

    Args:
        input_path: Path to input PNG file
        output_path: Path to output file (defaults to overwriting input)

    Returns:
        Path to optimized file
    """
    if not ZOPFLI_AVAILABLE:
        return input_path

    path = Path(input_path)
    output_path = output_path or str(path)

    try:
        with open(path, 'rb') as f:
            data = f.read()

        original_size = len(data)

        # Create ZopfliPNG compressor and optimize
        compressor = zopfli.ZopfliPNG()
        compressed = compressor.optimize(data)

        compressed_size = len(compressed)

        # Only write if we actually reduced size
        if compressed_size < original_size:
            with open(output_path, 'wb') as f:
                f.write(compressed)

            reduction = (1 - compressed_size / original_size) * 100
            sys.stderr.write(f"[ImageOptimizer] PNG optimized: {original_size} -> {compressed_size} bytes ({reduction:.1f}% reduction)\n")
        else:
            # If no reduction and output differs from input, copy original
            if output_path != str(path):
                with open(output_path, 'wb') as f:
                    f.write(data)
            sys.stderr.write(f"[ImageOptimizer] PNG already optimal, no changes made\n")

        return output_path

    except Exception as e:
        sys.stderr.write(f"[ImageOptimizer] PNG optimization failed: {e}\n")
        return input_path


def optimize_jpeg(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Optimize JPEG losslessly using MozJPEG.

    Args:
        input_path: Path to input JPEG file
        output_path: Path to output file (defaults to overwriting input)

    Returns:
        Path to optimized file
    """
    if not MOZJPEG_AVAILABLE:
        return input_path

    path = Path(input_path)
    output_path = output_path or str(path)

    try:
        with open(path, 'rb') as f:
            data = f.read()

        original_size = len(data)

        # Optimize with MozJPEG
        optimized = mozjpeg_optimize(data)

        optimized_size = len(optimized)

        # Only write if we actually reduced size
        if optimized_size < original_size:
            with open(output_path, 'wb') as f:
                f.write(optimized)

            reduction = (1 - optimized_size / original_size) * 100
            sys.stderr.write(f"[ImageOptimizer] JPEG optimized: {original_size} -> {optimized_size} bytes ({reduction:.1f}% reduction)\n")
        else:
            # If no reduction and output differs from input, copy original
            if output_path != str(path):
                with open(output_path, 'wb') as f:
                    f.write(data)
            sys.stderr.write(f"[ImageOptimizer] JPEG already optimal, no changes made\n")

        return output_path

    except Exception as e:
        sys.stderr.write(f"[ImageOptimizer] JPEG optimization failed: {e}\n")
        return input_path


def optimize_image(input_path: str, output_path: Optional[str] = None) -> str:
    """
    Auto-detect image format and optimize accordingly.

    Supports PNG (via Zopfli) and JPEG (via MozJPEG).
    Other formats are returned unchanged.

    Args:
        input_path: Path to input image file
        output_path: Path to output file (defaults to overwriting input)

    Returns:
        Path to optimized file (or original if format not supported)
    """
    path = Path(input_path)
    ext = path.suffix.lower()

    if ext == '.png':
        return optimize_png(input_path, output_path)
    elif ext in ('.jpg', '.jpeg'):
        return optimize_jpeg(input_path, output_path)
    else:
        # Unsupported format, return as-is
        return input_path


def is_available() -> dict:
    """
    Check which optimization backends are available.

    Returns:
        Dict with 'png' and 'jpeg' boolean keys
    """
    return {
        'png': ZOPFLI_AVAILABLE,
        'jpeg': MOZJPEG_AVAILABLE,
    }
