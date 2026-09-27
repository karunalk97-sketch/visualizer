"""Relief lighting: makes the flat, dithered picture read as a 3D surface.

The combined brightness (shapes + surf) is treated as a height map -- bright is
high, dark is low -- and lit from one side. Edges facing the light get a
highlight, edges facing away get a shadow, exactly like an embossed relief. It
is pure math on the picture, no extra layers, and it runs before dithering so
the highlights and shadows come out as dot density.
"""
from __future__ import annotations

import numpy as np

from .spectral_field import resize_bilinear

LIGHT = (-0.62, -0.78)   # (x, y): light coming from the upper left


def relief(intensity: np.ndarray, amount: float, light: tuple[float, float] = LIGHT) -> np.ndarray:
    """Returns the intensity with relief shading added. amount 0 = unchanged."""
    if amount <= 0.0:
        return intensity
    h, w = intensity.shape
    sh, sw = max(4, h // 2), max(4, w // 2)          # shade on a coarser grid: smoother and cheaper
    small = resize_bilinear(intensity, sh, sw)
    for _ in range(2):                                # soften, so slopes are gentle rather than jagged
        small = (4.0 * small + np.roll(small, 1, 0) + np.roll(small, -1, 0) + np.roll(small, 1, 1) + np.roll(small, -1, 1)) / 8.0
    gy, gx = np.gradient(small)
    shade = -(gx * light[0] + gy * light[1])          # slopes facing the light are brighter
    shade = shade * (2.6 * amount * sw / 110.0)       # keep the strength steady across resolutions
    shade = np.clip(shade, -0.45 * amount - 0.05, 0.45 * amount + 0.05)
    return np.clip(intensity + resize_bilinear(shade, h, w), 0.0, 1.0)
