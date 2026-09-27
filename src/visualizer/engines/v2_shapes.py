"""V2 -- Shapes. Every frequency owns random spots drawn as spinning shapes -- smooth
orbs, spiky orbs, long-spiked stars, lumpy blobs and a few long slashes. Each band is
judged against its own recent peak (a hi-hat lights as readily as a kick), bass is
grouped into a few big shapes, an area budget keeps a kick from filling the screen,
overlaps invert, and relief lighting gives it 3D depth."""
from __future__ import annotations

import numpy as np

from ..imaging import resize_bilinear
from .base import Engine, Inputs
from .relief import relief

ORB, SPIKY, STAR, BLOB, SLASH = range(5)
KIND_KEYS = ("orb", "spiky", "star", "blob", "slash")
_KIND_P = [0.22, 0.24, 0.26, 0.18, 0.10]
_BASS_FRAC = 0.22
_MID_FRAC = 0.50


def scatter(rng: np.random.Generator, n: int, candidates: int = 8) -> np.ndarray:
    pts = np.empty((n, 2), dtype=np.float32)
    pts[0] = rng.random(2)
    for i in range(1, n):
        cand = rng.random((candidates, 2)).astype(np.float32)
        d = ((cand[:, None, :] - pts[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        pts[i] = cand[int(np.argmax(d))]
    return pts


class ShapesField:
    """The V2 field as it was, plus a per-band gain (mix) and per-kind on/off."""

    NUM_KINDS = 5

    def __init__(self, num_bins: int, cluster_w: int, cluster_h: int, persistence: float = 0.88,
                 spots_per_bin: int = 3, drift: float = 0.03, contrast: float = 0.5, invert: float = 1.0,
                 max_active: int = 36, size: float = 0.6, area_budget: float = 0.08, max_bins: float = 0.18,
                 seed: int | None = None) -> None:
        self.num_bins = num_bins
        self.contrast = contrast
        self.invert = invert
        self.max_active = max_active
        self.size = size
        self.area_budget = area_budget
        self.max_bins = max_bins
        self.persistence = persistence
        self.drift = drift
        self._rng = np.random.default_rng(seed)
        frac = (np.arange(num_bins) + 0.5) / num_bins
        self._frac = frac.astype(np.float32)
        self._spots_per = np.where(frac < _BASS_FRAC, 1, np.where(frac < _MID_FRAC, 2, max(1, spots_per_bin) + 1))
        self._spots = int(self._spots_per.sum())
        self._bass_bins = int((frac < _BASS_FRAC).sum())
        self._bin_ref = np.zeros(num_bins, dtype=np.float32)
        self._frame = 0
        self._pos = scatter(self._rng, self._spots)
        self._target = self._pos.copy()
        self._set_owner()
        self._weight = self._rng.uniform(0.75, 1.0, self._spots).astype(np.float32)
        self._drift_rate = self._rng.uniform(0.004, 0.016, (self._spots, 2)).astype(np.float32)
        self._drift_phase = self._rng.uniform(0, 2 * np.pi, (self._spots, 2)).astype(np.float32)
        self._new_shapes()
        self.kinds_on = np.ones(self.NUM_KINDS, dtype=bool)
        self.resize(cluster_w, cluster_h)

    # -- layout ---------------------------------------------------------------

    def _set_owner(self) -> None:
        owner = np.repeat(np.arange(self.num_bins), self._spots_per)
        self._rng.shuffle(owner)
        self._owner = owner
        self._size_scale = (1.6 - 0.85 * self._frac[owner]).astype(np.float32)

    def _new_shapes(self) -> None:
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
        self._expo = np.where(kind == BLOB, 2.0, 3.0).astype(np.float32)
        self._angle = rng.uniform(0, np.pi, n).astype(np.float32)
        self._spin = (rng.choice([-1.0, 1.0], n) * u(0.008, 0.026)).astype(np.float32)

    def reshuffle(self) -> None:
        self._target = scatter(self._rng, self._spots)
        self._set_owner()
        self._drift_phase = self._rng.uniform(0, 2 * np.pi, (self._spots, 2)).astype(np.float32)
        self._new_shapes()

    def resize(self, cluster_w: int, cluster_h: int) -> None:
        self.cluster_w, self.cluster_h = cluster_w, cluster_h
        self.buffer = np.zeros((cluster_h, cluster_w), dtype=np.float32)

    def positions(self) -> np.ndarray:
        wobble = np.sin(self._frame * self._drift_rate + self._drift_phase) * self.drift
        p = np.clip(self._pos + wobble, 0.0, 1.0)
        return p * np.array([self.cluster_w - 1, self.cluster_h - 1], dtype=np.float32)

    # -- per frame ------------------------------------------------------------

    def _levels(self, band_levels: np.ndarray, gain: np.ndarray | None) -> np.ndarray:
        bins = self._shape(band_levels)
        if gain is not None:
            bins = np.clip(bins * gain, 0.0, 1.0)
        levels = bins[self._owner] * self._weight
        levels[~self.kinds_on[self._kind]] = 0.0                    # switched-off shapes never draw
        return levels

    def update(self, band_levels: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
        self._frame += 1
        self._pos += (self._target - self._pos) * 0.04
        levels = self._levels(band_levels, gain)
        self._draw(levels)
        return self.buffer

    def _draw(self, levels: np.ndarray) -> None:
        xy = self.positions()
        unit = max(1.0, min(self.cluster_w, self.cluster_h) / 22.0)
        layer = np.zeros_like(self.buffer)
        active = np.nonzero(levels > 0.05)[0]
        if len(active):
            active = active[np.argsort(levels[active])[::-1]]
            radii = unit * self.size * self._size_scale[active] * (0.8 + 1.5 * levels[active])
            area = (np.pi * radii ** 2 * np.sqrt(self._elong[active]) * (1.0 + 0.35 * self._depth[active]))
            keep = self._keep(area, active)
            active, radii = active[keep], radii[keep]
            for i, radius in zip(active, radii):
                self._splat(layer, i, xy[i, 0], xy[i, 1], float(radius), float(levels[i]))
        np.clip(layer, 0.0, 1.0, out=layer)
        np.maximum(self.buffer * self.persistence, layer, out=self.buffer)

    def _keep(self, area: np.ndarray, active: np.ndarray) -> np.ndarray:
        keep = np.cumsum(area) <= self.area_budget * self.cluster_w * self.cluster_h
        keep[0] = True
        keep[self.max_active:] = False
        return keep

    def _shape(self, band_levels: np.ndarray) -> np.ndarray:
        lv = np.asarray(band_levels, dtype=np.float32)
        if float(lv.max()) < 0.03:
            self._bin_ref *= 0.995
            return np.zeros_like(lv)
        self._bin_ref = np.maximum(lv, self._bin_ref * 0.9985)
        rel = lv / np.maximum(self._bin_ref, 0.045)
        nb = self._bass_bins - self._bass_bins % 4
        if nb:
            rel[:nb] = np.repeat(rel[:nb].reshape(-1, 4).max(axis=1), 4)
        score = rel * np.sqrt(lv / float(lv.max()))
        if nb:
            score[:nb] = np.repeat(score[:nb].reshape(-1, 4).max(axis=1), 4)
        k = max(3, int(round(self.max_bins * self.num_bins)))
        cut = float(np.partition(score, self.num_bins - k)[self.num_bins - k])
        keep = (score >= cut) & (rel > self.contrast)
        return np.where(keep, np.clip((rel - self.contrast) / (1.0 - self.contrast), 0.0, 1.0), 0.0).astype(np.float32)

    def _splat(self, layer: np.ndarray, i: int, cx: float, cy: float, radius: float, level: float) -> None:
        elong = float(self._elong[i])
        depth = float(self._depth[i]) * (0.6 + 0.8 * level)
        reach = min(radius * elong * (1.0 + depth), 0.45 * max(self.cluster_w, self.cluster_h))
        x0, x1 = max(0, int(cx - reach)), min(self.cluster_w, int(cx + reach) + 1)
        y0, y1 = max(0, int(cy - reach)), min(self.cluster_h, int(cy + reach) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        dx = (np.arange(x0, x1, dtype=np.float32)[None, :] - np.float32(cx))
        dy = (np.arange(y0, y1, dtype=np.float32)[:, None] - np.float32(cy))
        ang = float(self._angle[i]) + self._frame * float(self._spin[i])
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = dy * np.cos(ang) - dx * np.sin(ang)
        rho = np.sqrt((u / elong) ** 2 + (v * np.sqrt(elong)) ** 2) / max(radius, 1e-3)
        spikes = int(self._spikes[i])
        if spikes:
            phi = np.arctan2(v, u)
            edge = 1.0 + depth * np.abs(np.cos(spikes * phi / 2.0)) ** float(self._expo[i])
        else:
            edge = 1.0 + depth
        amp = np.clip((1.0 - rho / edge) * 1.8, 0.0, 1.0) * min(1.0, level * 1.3)
        sub = layer[y0:y1, x0:x1]
        layer[y0:y1, x0:x1] = sub + amp - 2.0 * self.invert * sub * amp


class ShapesEngine(Engine):
    KEY = "v2"
    LABEL = "V2 Shapes"
    ELEMENTS = (("orb", "Orbs"), ("spiky", "Spiky orbs"), ("star", "Stars"), ("blob", "Blobs"),
                ("slash", "Slashes"), ("depth", "3D depth"))
    FIELD = ShapesField
    CLUSTER = 3

    def __init__(self, width: int, height: int, seed: int | None = None) -> None:
        super().__init__(width, height, seed)
        self.field = self.FIELD(96, *self._cluster(), seed=int(self.rng.integers(0, 2**31 - 1)))

    def _cluster(self) -> tuple[int, int]:
        return max(16, self.W // self.CLUSTER), max(9, self.H // self.CLUSTER)

    def resize(self, width: int, height: int) -> None:
        super().resize(width, height)
        self.field.resize(*self._cluster())

    def reshuffle(self) -> None:
        self.field.reshuffle()

    def kind_keys(self) -> tuple[str, ...]:
        return KIND_KEYS

    def update(self, inp: Inputs, dt: float) -> np.ndarray:
        self.field.invert = self.invert
        self.field.kinds_on = np.array([self.on(k) for k in self.kind_keys()], dtype=bool)
        buf = self.field.update(inp.band_levels, inp.bin_gain)
        img = resize_bilinear(buf, self.H, self.W)
        return relief(img, 0.35) if self.on("depth") else img
