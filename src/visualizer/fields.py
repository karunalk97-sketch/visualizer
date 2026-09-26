"""Builds the full-screen intensity field for the halftone visualization mode:
every column maps to its own frequency band (so "every frequency is mapped
differently" across the width of the screen), bars rise from the bottom, and
a slow animated texture keeps the *whole* field alive (not just the bars)
scaled by overall loudness, so quiet audio still breathes and loud audio
blooms outward.
"""
from __future__ import annotations

import numpy as np


def build_field(band_levels: np.ndarray, grid_w: int, grid_h: int, t: float) -> np.ndarray:
    """Returns a (grid_h, grid_w) float32 array in [0, 1]."""
    n = len(band_levels)
    col_positions = np.linspace(0, n - 1, grid_w)
    cols = np.interp(col_positions, np.arange(n), band_levels)  # (grid_w,)

    rows = np.arange(grid_h)
    row_frac = rows / max(1, grid_h - 1)  # 0 at top, 1 at bottom
    # bars fill from the bottom row upward: a row is lit once column energy
    # reaches "how far from the bottom" that row is.
    bar_mask = (row_frac[:, None] >= (1.0 - cols[None, :])).astype(np.float32)  # (h, w)

    xx, yy = np.meshgrid(np.linspace(0, 1, grid_w), np.linspace(0, 1, grid_h))
    texture = 0.5 + 0.5 * np.sin(2 * np.pi * (xx * 3.0 + t * 0.05)) * np.sin(
        2 * np.pi * (yy * 2.0 - t * 0.07)
    )
    overall = float(np.mean(band_levels))

    intensity = bar_mask * 0.85 + texture * (0.15 + 0.35 * overall)
    return np.clip(intensity, 0.0, 1.0).astype(np.float32)
