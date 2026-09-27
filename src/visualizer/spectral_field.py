"""The core visual mechanism: every tracked frequency bin owns a few *random*
spots on the canvas. A spot only lights up when its frequency actually has
energy, so which spots glow -- and how big they get -- is a direct trace of the
music. Nothing is laid out in frequency order: neighbouring frequencies land in
unrelated places, so there is no left-to-right sweep, no spiral, no symmetry.

Each frequency is judged against *its own* recent loudness, not against the
loudest sound in the whole spectrum, so a hi-hat is as able to light its spots
as a kick is. Bass gets few, large shapes (grouped, so one kick is a handful of
big shapes rather than dozens); treble gets many small ones. A per-frame area
budget stops any single hit from lighting the whole screen.

Spots come in five shapes -- a smooth orb, a spiky orb, a long-spiked star, an
ambiguous lumpy blob, and (a small share) a long thin slash -- each with its own
angle, spinning steadily. Within a frame, overlapping shapes XOR: they invert each other
instead of just getting brighter.

To keep it alive rather than static:

* every spot wanders slowly on its own smooth path (organic drift);
* `reshuffle()` draws a brand new random layout -- the spots glide to their new
  places instead of jumping -- and the app calls it on every new song.

Positions are stored normalised (0..1) so the layout survives window resizes.
"""
from __future__ import annotations

import numpy as np

ORB, SPIKY, STAR, BLOB, SLASH = range(5)
_KIND_P = [0.22, 0.24, 0.26, 0.18, 0.10]   # slashes are the long thin ones: keep them a small share
_BASS_FRAC = 0.22        # lowest ~22% of the (log) frequency axis: kicks, sub, bass
_MID_FRAC = 0.50


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
        size: float = 0.6,
        area_budget: float = 0.08,
        max_bins: float = 0.18,
        seed: int | None = None,
    ) -> None:
        self.num_bins = num_bins
        self.contrast = contrast
        self.invert = invert  # 0 = overlaps just add, 1 = overlaps fully invert
        self.max_active = max_active
        self.size = size  # overall shape scale
        self.area_budget = area_budget  # most of the canvas one frame's shapes may claim
        self.max_bins = max_bins        # most of the frequency bands that may light at once
        self.persistence = persistence
        self.drift = drift
        self._rng = np.random.default_rng(seed)
        frac = (np.arange(num_bins) + 0.5) / num_bins
        self._frac = frac.astype(np.float32)
        # bass: 1 large spot per bin; mids: 2; treble: more, smaller ones
        self._spots_per = np.where(frac < _BASS_FRAC, 1, np.where(frac < _MID_FRAC, 2, max(1, spots_per_bin) + 1))
        self._spots = int(self._spots_per.sum())
        self._bass_bins = int((frac < _BASS_FRAC).sum())
        self._bin_ref = np.zeros(num_bins, dtype=np.float32)
        self._frame = 0

        self._pos = _scatter(self._rng, self._spots)
        self._target = self._pos.copy()
        self._set_owner()
        self._weight = self._rng.uniform(0.75, 1.0, self._spots).astype(np.float32)
        # each spot drifts on its own slow, smooth loop
        self._drift_rate = self._rng.uniform(0.004, 0.016, (self._spots, 2)).astype(np.float32)
        self._drift_phase = self._rng.uniform(0, 2 * np.pi, (self._spots, 2)).astype(np.float32)
        self._new_shapes()

        self.resize(cluster_w, cluster_h)

    # -- layout ---------------------------------------------------------------

    def _set_owner(self) -> None:
        owner = np.repeat(np.arange(self.num_bins), self._spots_per)
        self._rng.shuffle(owner)
        self._owner = owner
        # low frequencies draw big, high frequencies small
        self._size_scale = (1.6 - 0.85 * self._frac[owner]).astype(np.float32)

    def _new_shapes(self) -> None:
        """Give every spot its own shape and a steady spin (they all visibly rotate)."""
        rng, n = self._rng, self._spots
        kind = rng.choice(5, size=n, p=_KIND_P)
        self._kind = kind
        u = lambda lo, hi: rng.uniform(lo, hi, n)  # noqa: E731
        conds = [kind == ORB, kind == SPIKY, kind == STAR, kind == BLOB]
        self._elong = np.select(conds, [u(1.0, 1.1), u(1.0, 1.1), u(1.0, 1.25), u(1.0, 1.3)], u(2.0, 3.2)).astype(np.float32)
        self._spikes = np.select(
            conds, [np.zeros(n), rng.integers(11, 19, n), rng.integers(5, 10, n), rng.integers(2, 4, n)], rng.integers(0, 3, n),
        ).astype(np.int32)
        self._depth = np.select(conds, [np.zeros(n), u(0.35, 0.6), u(1.3, 2.4), u(0.25, 0.55)], u(0.2, 0.5)).astype(np.float32)
        # sharpness of the spikes: a blob's lobes are smooth (cos^2, no cusps); spikes are sharp
        self._expo = np.where(kind == BLOB, 2.0, 3.0).astype(np.float32)
        self._angle = rng.uniform(0, np.pi, n).astype(np.float32)
        self._spin = (rng.choice([-1.0, 1.0], n) * u(0.008, 0.026)).astype(np.float32)   # 0.5-1.5 rad/s, either way
    def reshuffle(self) -> None:
        """Draw a brand new random layout; spots glide there over ~a second."""
        self._target = _scatter(self._rng, self._spots)
        self._set_owner()
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

        bins = self._shape(band_levels)
        levels = bins[self._owner] * self._weight
        xy = self.positions()
        unit = max(1.0, min(self.cluster_w, self.cluster_h) / 22.0)
        layer = np.zeros_like(self.buffer)  # this frame's shapes; overlaps invert inside it
        active = np.nonzero(levels > 0.05)[0]
        if len(active):
            active = active[np.argsort(levels[active])[::-1]]       # strongest first
            radii = unit * self.size * self._size_scale[active] * (0.8 + 1.5 * levels[active])
            # rough area each shape will claim, then keep the strongest that fit the budget
            area = (np.pi * radii ** 2 * np.sqrt(self._elong[active])
                    * (1.0 + 0.35 * self._depth[active]))
            keep = np.cumsum(area) <= self.area_budget * self.cluster_w * self.cluster_h
            keep[0] = True                                            # one always gets through
            keep[self.max_active:] = False
            active, radii = active[keep], radii[keep]
            for i, radius in zip(active, radii):
                self._splat(layer, i, xy[i, 0], xy[i, 1], float(radius), float(levels[i]))
        np.clip(layer, 0.0, 1.0, out=layer)
        # the glow trail fades on its own; it is not XORed, or a held note would strobe
        np.maximum(self.buffer * self.persistence, layer, out=self.buffer)
        return self.buffer

    def _shape(self, band_levels: np.ndarray) -> np.ndarray:
        """Per-bin levels 0..1. Every frequency is compared with its *own* recent
        peak (so treble competes fairly with bass), bass bins are grouped, and dense
        music gets a stricter gate so it doesn't light everything."""
        lv = np.asarray(band_levels, dtype=np.float32)
        if float(lv.max()) < 0.03:
            self._bin_ref *= 0.995
            return np.zeros_like(lv)
        self._bin_ref = np.maximum(lv, self._bin_ref * 0.9985)
        rel = lv / np.maximum(self._bin_ref, 0.045)      # 0..1: loud for THIS band
        nb = self._bass_bins - self._bass_bins % 4       # bass in groups of 4: one kick = a few big shapes
        if nb:
            rel[:nb] = np.repeat(rel[:nb].reshape(-1, 4).max(axis=1), 4)
        # Rank the bands: relative loudness weighted by real energy, so a kick's
        # broadband click can't outrank the bass, and a hi-hat wins the treble.
        score = rel * np.sqrt(lv / float(lv.max()))
        if nb:
            score[:nb] = np.repeat(score[:nb].reshape(-1, 4).max(axis=1), 4)   # a bass group lights together, or not at all
        k = max(3, int(round(self.max_bins * self.num_bins)))
        cut = float(np.partition(score, self.num_bins - k)[self.num_bins - k])   # k-th largest
        keep = (score >= cut) & (rel > self.contrast)
        return np.where(keep, np.clip((rel - self.contrast) / (1.0 - self.contrast), 0.0, 1.0), 0.0).astype(np.float32)

    def _splat(self, layer: np.ndarray, i: int, cx: float, cy: float, radius: float, level: float) -> None:
        """Draws spot i's shape into this frame's fresh layer. Overlaps XOR
        (a + b - 2ab), so where two shapes cross the pixels invert instead of
        just getting brighter."""
        elong = float(self._elong[i])
        kind = int(self._kind[i])
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
            edge = 1.0 + depth * np.abs(np.cos(spikes * phi / 2.0)) ** float(self._expo[i])
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
