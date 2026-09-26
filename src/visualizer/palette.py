"""Grayscale quantization: black and white only, at a chosen number of gray
levels (bit depth), with ordered (Bayer) dithering so low bit depths still
read as a dense dot pattern rather than flat, banded regions.
"""
from __future__ import annotations

import numpy as np

RGB = tuple[int, int, int]

# 4x4 Bayer matrix, normalized to 0..1, used for ordered dithering.
_BAYER_4X4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.float32,
) / 16.0


def grayscale_palette(bit_depth: int) -> list[RGB]:
    """bit_depth=1 -> pure black/white; bit_depth=N -> 2**N evenly spaced grays."""
    levels = 2 ** bit_depth
    return [(v, v, v) for v in (round(255 * i / (levels - 1)) for i in range(levels))]


def quantize(intensity: np.ndarray, colors: list[RGB]) -> np.ndarray:
    """Map a (H, W) float array in [0, 1] to an (H, W, 3) uint8 image using
    only the given (grayscale) palette, with ordered dithering.
    """
    n = len(colors)
    intensity = np.clip(intensity, 0.0, 1.0)

    if n > 1:
        h, w = intensity.shape
        tile = np.tile(_BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        step = 1.0 / (n - 1)
        intensity = intensity + (tile - 0.5) * step

    indices = np.clip(np.round(np.clip(intensity, 0.0, 1.0) * (n - 1)), 0, n - 1).astype(np.int32)
    palette_arr = np.array(colors, dtype=np.uint8)
    return palette_arr[indices]
