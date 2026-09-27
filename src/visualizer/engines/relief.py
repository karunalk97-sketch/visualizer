"""Relief lighting (the "3D depth" of V2/V3): the picture is treated as a height map
and lit from the upper left, like an embossed relief."""
from __future__ import annotations

import numpy as np

from ..imaging import resize_bilinear

LIGHT = (-0.62, -0.78)


def relief(intensity: np.ndarray, amount: float, light: tuple[float, float] = LIGHT) -> np.ndarray:
    if amount <= 0.0:
        return intensity
    h, w = intensity.shape
    sh, sw = max(4, h // 2), max(4, w // 2)
    small = resize_bilinear(intensity, sh, sw)
    for _ in range(2):
        small = (4.0 * small + np.roll(small, 1, 0) + np.roll(small, -1, 0) + np.roll(small, 1, 1) + np.roll(small, -1, 1)) / 8.0
    gy, gx = np.gradient(small)
    shade = -(gx * light[0] + gy * light[1])
    shade = shade * (2.6 * amount * sw / 110.0)
    shade = np.clip(shade, -0.45 * amount - 0.05, 0.45 * amount + 0.05)
    return np.clip(intensity + resize_bilinear(shade, h, w), 0.0, 1.0)
