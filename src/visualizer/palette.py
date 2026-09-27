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


_tile_cache: dict[tuple[int, int], np.ndarray] = {}


def _dither_tile(h: int, w: int) -> np.ndarray:
    """The Bayer pattern cropped to (h, w); cached because the grid rarely changes."""
    key = (h, w)
    if key not in _tile_cache:
        if len(_tile_cache) > 8:
            _tile_cache.clear()
        _tile_cache[key] = np.tile(_BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w].astype(np.float32)
    return _tile_cache[key]


def grayscale_palette(bit_depth: int) -> list[RGB]:
    """bit_depth=1 -> pure black/white; bit_depth=N -> 2**N evenly spaced grays."""
    levels = 2 ** bit_depth
    return [(v, v, v) for v in (round(255 * i / (levels - 1)) for i in range(levels))]


def quantize_indices(intensity: np.ndarray, n_colors: int) -> np.ndarray:
    """Map a (H, W) float array in [0, 1] to (H, W) uint8 palette indices with
    ordered dithering. This is the fast path: the renderer draws it as an 8-bit
    palettised surface."""
    h, w = intensity.shape
    x = np.clip(intensity, 0.0, 1.0) * (n_colors - 1)
    if n_colors > 1:
        # threshold each cell against its (centred) Bayer value: pure white stays pure
        # white and pure black stays black -- no stray dots in solid areas
        x = x + _dither_tile(h, w) + (0.5 / 16.0)
    else:
        x = x + 0.5
    np.floor(x, out=x)
    return np.clip(x, 0, n_colors - 1).astype(np.uint8)


def quantize(intensity: np.ndarray, colors: list[RGB]) -> np.ndarray:
    """Map a (H, W) float array in [0, 1] to an (H, W, 3) uint8 image using
    only the given (grayscale) palette, with ordered dithering.
    """
    indices = quantize_indices(intensity, len(colors))
    return np.array(colors, dtype=np.uint8)[indices]
