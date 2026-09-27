"""V1 -- Spots. The first version: every one of 96 frequency bands owns three random
spots; a spot glows when its frequency stands out, soft and additive, leaving a
glowing trail. A new random layout every song."""
from __future__ import annotations

import numpy as np

from .base import Engine, Inputs


def _scatter(rng: np.random.Generator, n: int, candidates: int = 8) -> np.ndarray:
    pts = np.empty((n, 2), dtype=np.float32)
    pts[0] = rng.random(2)
    for i in range(1, n):
        cand = rng.random((candidates, 2)).astype(np.float32)
        d = ((cand[:, None, :] - pts[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        pts[i] = cand[int(np.argmax(d))]
    return pts


def resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape
    rows = np.minimum((np.arange(out_h) * in_h) // out_h, in_h - 1)
    cols = np.minimum((np.arange(out_w) * in_w) // out_w, in_w - 1)
    return arr[rows][:, cols]


class _SpotField:
    """The original field (restored as it was), plus a per-band gain for the mix."""

    def __init__(self, num_bins: int, cluster_w: int, cluster_h: int, persistence: float = 0.88,
                 spots_per_bin: int = 3, drift: float = 0.03, contrast: float = 0.5, seed: int | None = None) -> None:
        self.num_bins = num_bins
        self.contrast = contrast
        self.persistence = persistence
        self.drift = drift
        self._rng = np.random.default_rng(seed)
        self._spots_per_bin = max(1, spots_per_bin)
        self._spots = num_bins * self._spots_per_bin
        self._frame = 0
        self._peak_ref = 0.0
        self._pos = _scatter(self._rng, self._spots)
        self._target = self._pos.copy()
        self._owner = self._new_owner()
        self._weight = self._rng.uniform(0.75, 1.0, self._spots).astype(np.float32)
        self._drift_rate = self._rng.uniform(0.004, 0.016, (self._spots, 2)).astype(np.float32)
        self._drift_phase = self._rng.uniform(0, 2 * np.pi, (self._spots, 2)).astype(np.float32)
        self.resize(cluster_w, cluster_h)

    def _new_owner(self) -> np.ndarray:
        owner = np.repeat(np.arange(self.num_bins), self._spots_per_bin)
        self._rng.shuffle(owner)
        return owner

    def reshuffle(self) -> None:
        self._target = _scatter(self._rng, self._spots)
        self._owner = self._new_owner()
        self._drift_phase = self._rng.uniform(0, 2 * np.pi, (self._spots, 2)).astype(np.float32)

    def resize(self, cluster_w: int, cluster_h: int) -> None:
        self.cluster_w, self.cluster_h = cluster_w, cluster_h
        self.buffer = np.zeros((cluster_h, cluster_w), dtype=np.float32)

    def positions(self) -> np.ndarray:
        wobble = np.sin(self._frame * self._drift_rate + self._drift_phase) * self.drift
        p = np.clip(self._pos + wobble, 0.0, 1.0)
        return p * np.array([self.cluster_w - 1, self.cluster_h - 1], dtype=np.float32)

    def update(self, band_levels: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
        self._frame += 1
        self._pos += (self._target - self._pos) * 0.04
        self.buffer *= self.persistence
        shaped = self._shape(band_levels)
        if gain is not None:
            shaped = np.clip(shaped * gain, 0.0, 1.0)
        levels = shaped[self._owner] * self._weight
        xy = self.positions()
        unit = max(1.0, min(self.cluster_w, self.cluster_h) / 22.0)
        for i in np.nonzero(levels > 0.05)[0]:
            level = float(levels[i])
            self._splat(xy[i, 0], xy[i, 1], unit * (0.9 + 2.6 * level), level)
        np.clip(self.buffer, 0.0, 1.0, out=self.buffer)
        return self.buffer

    def _shape(self, band_levels: np.ndarray) -> np.ndarray:
        peak = float(band_levels.max())
        self._peak_ref = max(peak, self._peak_ref * 0.995)
        ref = self._peak_ref
        if ref < 0.03:
            return band_levels * 0.0
        gate = self.contrast * ref
        shaped = np.clip((band_levels - gate) / max(ref - gate, 1e-6), 0.0, 1.0)
        return shaped * min(1.0, ref / 0.2)

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


class SpotsEngine(Engine):
    KEY = "v1"
    LABEL = "V1 Spots"
    ELEMENTS = (("spots", "Spots"), ("glow", "Glow trail"), ("drift", "Drift"))
    CLUSTER = 3                   # the field runs 3 cells coarser, then is scaled up with chunky pixels

    def __init__(self, width: int, height: int, seed: int | None = None) -> None:
        super().__init__(width, height, seed)
        self.field = _SpotField(96, *self._cluster(), seed=int(self.rng.integers(0, 2**31 - 1)))

    def _cluster(self) -> tuple[int, int]:
        return max(8, self.W // self.CLUSTER), max(8, self.H // self.CLUSTER)

    def resize(self, width: int, height: int) -> None:
        super().resize(width, height)
        self.field.resize(*self._cluster())

    def reshuffle(self) -> None:
        self.field.reshuffle()

    def update(self, inp: Inputs, dt: float) -> np.ndarray:
        self.field.persistence = 0.88 if self.on("glow") else 0.0
        self.field.drift = 0.03 if self.on("drift") else 0.0
        gain = inp.bin_gain if self.on("spots") else np.zeros_like(inp.bin_gain)
        return resize_nearest(self.field.update(inp.band_levels, gain), self.H, self.W)
