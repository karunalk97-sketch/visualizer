"""The core visual mechanism: every tracked frequency bin owns a few *random*
spots on the canvas. A spot only lights up when its frequency actually has
energy, so which spots glow -- and how big they get -- is a direct trace of the
music. Nothing is laid out in frequency order: neighbouring frequencies land in
unrelated places, so there is no left-to-right sweep, no spiral, no symmetry.

Each spot also has its own shape (blob, spiky star, long string, or a stretched
spiky hybrid) and slowly spins. Within a frame, overlapping shapes XOR -- they
invert each other instead of just getting brighter.

To keep it alive rather than static:

* every spot wanders slowly on its own smooth path (organic drift);
* `reshuffle()` draws a brand new random layout -- the spots glide to their new
  places instead of jumping -- and the app calls it on every new song.

Positions are stored normalised (0..1) so the layout survives window resizes.
"""
from __future__ import annotations

import numpy as np


def _scatter(rng: np.random.Generator, n: int, candidates: int = 8) -> np.ndarray:
    """n random points in the unit square via best-candidate sampling: each new
    point is the farthest of a few random candidates from the points so far.
    Looks random (no lattice, no symmetry) but avoids big empty patches and
    pile-ups, so the field fills the screen evenly."""
    pts = np.empty((n, 2), dtype=np.float32)
    pts[0] = rng.random(2)
    for i in range(1, n):
        cand = rng.random((candidates, 2)).astype(np.float32)
        d = ((cand[:, None, :] - pts[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        pts[i] = cand[int(np.argmax(d))]
    return pts


class SpectralField:
    def __init__(
        self,
        num_bins: int,
        cluster_w: int,
        cluster_h: int,
        persistence: float = 0.88,
        spots_per_bin: int = 3,
        drift: float = 0.03,
        contrast: float = 0.5,
        invert: float = 0.85,
        max_active: int = 36,
        size: float = 0.65,
        seed: int | None = None,
    ) -> None:
        self.num_bins = num_bins
        self.contrast = contrast
        self.invert = invert  # 0 = overlaps just add, 1 = overlaps fully invert
        self.max_active = max_active
        self.size = size  # overall shape scale
        self.persistence = persistence
        self.drift = drift
        self._rng = np.random.default_rng(seed)
        self._spots = num_bins * max(1, spots_per_bin)
        self._spots_per_bin = max(1, spots_per_bin)
        self._frame = 0

        self._pos = _scatter(self._rng, self._spots)
        self._target = self._pos.copy()
        self._owner = self._new_owner()
        self._weight = self._rng.uniform(0.75, 1.0, self._spots).astype(np.float32)
        # each spot drifts on its own slow, smooth loop
        self._drift_rate = self._rng.uniform(0.004, 0.016, (self._spots, 2)).astype(np.float32)
        self._drift_phase = self._rng.uniform(0, 2 * np.pi, (self._spots, 2)).astype(np.float32)
        self._new_shapes()

        self.resize(cluster_w, cluster_h)

    # -- layout ---------------------------------------------------------------

    def _new_shapes(self) -> None:
        """Give every spot its own shape: round blob, spiky star, long string, or
        a stretched spiky hybrid -- each with its own angle and slow spin."""
        rng, n = self._rng, self._spots
        kind = rng.choice(4, size=n, p=[0.28, 0.28, 0.26, 0.18])
        self._kind = kind
        self._elong = np.select(
            [kind == 0, kind == 1, kind == 2],
            [rng.uniform(1.0, 1.3, n), rng.uniform(1.0, 1.4, n), rng.uniform(2.6, 5.0, n)],
            rng.uniform(1.8, 3.0, n),
        ).astype(np.float32)
        self._spikes = np.select(
            [kind == 0, kind == 1, kind == 2],
            [np.zeros(n), rng.integers(5, 11, n), rng.integers(0, 3, n)],
            rng.integers(3, 7, n),
        ).astype(np.int32)
        self._depth = np.select(
            [kind == 0, kind == 1, kind == 2],
            [np.zeros(n), rng.uniform(1.3, 2.4, n), rng.uniform(0.2, 0.5, n)],
            rng.uniform(0.8, 1.5, n),
        ).astype(np.float32)
        self._angle = rng.uniform(0, np.pi, n).astype(np.float32)
        self._spin = rng.uniform(-0.012, 0.012, n).astype(np.float32)

    def _new_owner(self) -> np.ndarray:
        owner = np.repeat(np.arange(self.num_bins), self._spots_per_bin)
        self._rng.shuffle(owner)
        return owner

    def reshuffle(self) -> None:
        """Draw a brand new random layout; spots glide there over ~a second."""
        self._target = _scatter(self._rng, self._spots)
        self._owner = self._new_owner()
        self._drift_phase = self._rng.uniform(0, 2 * np.pi, (self._spots, 2)).astype(np.float32)
        self._new_shapes()

    def resize(self, cluster_w: int, cluster_h: int) -> None:
        self.cluster_w = cluster_w
        self.cluster_h = cluster_h
        self.buffer = np.zeros((cluster_h, cluster_w), dtype=np.float32)

    def positions(self) -> np.ndarray:
        """Current spot positions in cluster-grid units, shape (spots, 2) as (x, y)."""
        wobble = np.sin(self._frame * self._drift_rate + self._drift_phase) * self.drift
        p = np.clip(self._pos + wobble, 0.0, 1.0)
        return p * np.array([self.cluster_w - 1, self.cluster_h - 1], dtype=np.float32)

    # -- per frame ------------------------------------------------------------

    def update(self, band_levels: np.ndarray) -> np.ndarray:
        self._frame += 1
        self._pos += (self._target - self._pos) * 0.04  # glide toward the current layout

        band_levels = self._shape(band_levels)
        levels = band_levels[self._owner] * self._weight
        xy = self.positions()
        unit = max(1.0, min(self.cluster_w, self.cluster_h) / 22.0)
        layer = np.zeros_like(self.buffer)  # this frame's shapes; overlaps invert inside it
        active = np.nonzero(levels > 0.05)[0]
        if len(active) > self.max_active:  # keep the strongest; bounds the per-frame cost
            active = active[np.argsort(levels[active])[-self.max_active:]]
        for i in active:
            level = float(levels[i])
            self._splat(layer, i, xy[i, 0], xy[i, 1], unit * self.size * (0.8 + 1.5 * level), level)
        np.clip(layer, 0.0, 1.0, out=layer)
        # the glow trail fades on its own; it is not XORed, or a held note would strobe
        np.maximum(self.buffer * self.persistence, layer, out=self.buffer)
        return self.buffer

    def _shape(self, band_levels: np.ndarray) -> np.ndarray:
        """Contrast: dense music has energy in almost every bin, which would
        light the whole screen. Keep only what stands out against the current
        peak, and scale by overall loudness so silence stays dark."""
        peak = float(band_levels.max())
        self._peak_ref = max(peak, getattr(self, "_peak_ref", 0.0) * 0.995)
        ref = self._peak_ref
        if ref < 0.03:
            return band_levels * 0.0
        # dense music has most bins near the peak: tighten the gate as it gets denser
        crowded = float((band_levels > 0.5 * ref).mean())
        gate = min(0.85, self.contrast + 1.0 * max(0.0, crowded - 0.2)) * ref
        shaped = np.clip((band_levels - gate) / max(ref - gate, 1e-6), 0.0, 1.0)
        return shaped * min(1.0, ref / 0.2)

    def _splat(self, layer: np.ndarray, i: int, cx: float, cy: float, radius: float, level: float) -> None:
        """Draws spot i's shape into this frame's fresh layer. Overlaps XOR
        (a + b - 2ab), so where two shapes cross the pixels invert instead of
        just getting brighter."""
        elong = float(self._elong[i])
        depth = float(self._depth[i]) * (0.6 + 0.8 * level)  # louder -> spikier
        reach = min(radius * elong * (1.0 + depth), 0.45 * max(self.cluster_w, self.cluster_h))
        x0, x1 = max(0, int(cx - reach)), min(self.cluster_w, int(cx + reach) + 1)
        y0, y1 = max(0, int(cy - reach)), min(self.cluster_h, int(cy + reach) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        dx = (np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(cx))
        dy = (np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(cy))

        ang = float(self._angle[i]) + self._frame * float(self._spin[i])
        u = dx * np.cos(ang) + dy * np.sin(ang)   # along the shape's axis
        v = dy * np.cos(ang) - dx * np.sin(ang)   # across it
        rho = np.sqrt((u / elong) ** 2 + (v * np.sqrt(elong)) ** 2) / max(radius, 1e-3)

        spikes = int(self._spikes[i])
        if spikes:
            phi = np.arctan2(v, u)
            edge = 1.0 + depth * np.abs(np.cos(spikes * phi / 2.0)) ** 3
        else:
            edge = 1.0 + depth
        # solid core with a soft rim, so overlaps read as a clear negative
        amp = np.clip((1.0 - rho / edge) * 1.8, 0.0, 1.0) * min(1.0, level * 1.3)

        sub = layer[y0:y1, x0:x1]
        layer[y0:y1, x0:x1] = sub + amp - 2.0 * self.invert * sub * amp


_resize_cache: dict[tuple[int, int, int, int], tuple] = {}


def _resize_plan(in_h: int, in_w: int, out_h: int, out_w: int) -> tuple:
    key = (in_h, in_w, out_h, out_w)
    if key not in _resize_cache:
        if len(_resize_cache) > 16:
            _resize_cache.clear()
        ys = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * in_h / out_h - 0.5, 0, in_h - 1)
        xs = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * in_w / out_w - 0.5, 0, in_w - 1)
        y0 = np.floor(ys).astype(np.intp)
        x0 = np.floor(xs).astype(np.intp)
        _resize_cache[key] = (
            y0, np.minimum(y0 + 1, in_h - 1), x0, np.minimum(x0 + 1, in_w - 1),
            (ys - y0).astype(np.float32)[:, None], (xs - x0).astype(np.float32)[None, :],
        )
    return _resize_cache[key]


def resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Smooth upscale, so shapes stay smooth however fine the pixel grid is.
    Separable (columns first, on the small array) with cached index tables."""
    in_h, in_w = arr.shape
    y0, y1, x0, x1, wy, wx = _resize_plan(in_h, in_w, out_h, out_w)
    arr = arr.astype(np.float32, copy=False)
    wide = arr[:, x0] * (1.0 - wx) + arr[:, x1] * wx          # (in_h, out_w) -- still small
    out = wide[y0]
    out *= 1.0 - wy
    out += wide[y1] * wy
    return out


def resize_nearest(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    in_h, in_w = arr.shape
    row_idx = np.minimum((np.arange(out_h) * in_h) // out_h, in_h - 1)
    col_idx = np.minimum((np.arange(out_w) * in_w) // out_w, in_w - 1)
    return arr[row_idx][:, col_idx]
