"""The core visual mechanism: every tracked frequency bin owns a fixed spot
on screen along a multi-turn spiral (not a grid, not mirrored), and that
spot densifies only when that frequency actually has energy. Bins are laid
out *in frequency order* along the spiral (with only small jitter) so that
a single note -- which always spreads its energy across a contiguous run of
neighboring bins via harmonics and FFT smearing -- lights up one connected
regional cluster, not scattered specks all over the canvas. Which regions
light up, and how large the clusters get, depends entirely on which
frequencies are actually active, so a bass-heavy track, a bright treble
track, and a full mix each carve out a different shape; nothing here is a
decorative texture or a fixed bar/level meter.
"""
from __future__ import annotations

import numpy as np


class SpectralField:
    def __init__(
        self,
        num_bins: int,
        cluster_w: int,
        cluster_h: int,
        persistence: float = 0.88,
        revolutions: float = 3.2,
        seed: int = 7,
    ) -> None:
        self.cluster_w = cluster_w
        self.cluster_h = cluster_h
        self.persistence = persistence
        self.buffer = np.zeros((cluster_h, cluster_w), dtype=np.float32)

        rng = np.random.default_rng(seed)
        frac = (np.arange(num_bins) + 0.5) / num_bins  # 0..1 in frequency order
        r = np.sqrt(frac)  # sqrt spacing -> ~uniform area density per turn of the spiral
        theta = frac * revolutions * 2.0 * np.pi

        # small jitter for an organic, hand-drawn feel -- kept well below the
        # spacing between neighboring bins so frequency locality is preserved
        r = np.clip(r + rng.uniform(-0.02, 0.02, num_bins), 0.02, 1.0)
        theta = theta + rng.uniform(-0.05, 0.05, num_bins)

        self._bx = (0.5 + 0.48 * r * np.cos(theta)) * (cluster_w - 1)
        self._by = (0.5 + 0.48 * r * np.sin(theta)) * (cluster_h - 1)
        self._base_radius = 1.0 + 2.0 * r  # bins further out get a touch more natural spread

    def update(self, band_levels: np.ndarray) -> np.ndarray:
        self.buffer *= self.persistence
        for idx, level in enumerate(band_levels):
            if level <= 0.02:
                continue
            self._splat(self._bx[idx], self._by[idx], self._base_radius[idx] + level * 4.0, level)
        np.clip(self.buffer, 0.0, 1.0, out=self.buffer)
        return self.buffer

    def _splat(self, cx: float, cy: float, radius: float, level: float) -> None:
        x0, x1 = max(0, int(cx - radius)), min(self.cluster_w, int(cx + radius) + 1)
        y0, y1 = max(0, int(cy - radius)), min(self.cluster_h, int(cy + radius) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        dx = np.arange(x0, x1)[None, :] - cx
        dy = np.arange(y0, y1)[:, None] - cy
        dist = np.sqrt(dx * dx + dy * dy)
        kernel = np.clip(1.0 - dist / max(radius, 1e-3), 0.0, 1.0) ** 1.5
        self.buffer[y0:y1, x0:x1] += kernel * level


def resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape
    row_idx = np.minimum((np.arange(out_h) * in_h) // out_h, in_h - 1)
    col_idx = np.minimum((np.arange(out_w) * in_w) // out_w, in_w - 1)
    return arr[row_idx][:, col_idx]
