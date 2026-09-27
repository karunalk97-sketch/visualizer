"""V3 -- Adaptive shapes. V2's field, but a spot re-rolls its shape every time it
starts to sound, weighted by how loud the moment is: quiet passages draw soft orbs,
blobs and wisps; loud ones draw spiky orbs, stars and jagged shards. Each frequency
range keeps an identity (bass always round and heavy, treble thin and sparkly), bold
and soft shapes get separate shares of the screen, and quiet detail keeps out of
the way of a loud hit."""
from __future__ import annotations

import numpy as np

from .v2_shapes import ShapesEngine, ShapesField, scatter

ORB, SPIKY, STAR, BLOB, SLASH, WISP, SHARD = range(7)
KIND_KEYS = ("orb", "spiky", "star", "blob", "slash", "wisp", "shard")
BOLD = (SPIKY, STAR, SHARD)
_BREAKS = np.array([0.22, 0.42, 0.60, 0.80], dtype=np.float32)
_QUIET = np.array([
    [0.42, 0.00, 0.00, 0.43, 0.00, 0.15, 0.00],
    [0.22, 0.00, 0.00, 0.40, 0.13, 0.25, 0.00],
    [0.15, 0.05, 0.30, 0.20, 0.05, 0.25, 0.00],
    [0.15, 0.25, 0.10, 0.15, 0.10, 0.20, 0.05],
    [0.10, 0.20, 0.00, 0.05, 0.35, 0.25, 0.05],
])
_LOUD = np.array([
    [0.48, 0.00, 0.00, 0.47, 0.00, 0.05, 0.00],
    [0.12, 0.18, 0.00, 0.35, 0.10, 0.10, 0.15],
    [0.05, 0.15, 0.45, 0.10, 0.05, 0.05, 0.15],
    [0.05, 0.30, 0.10, 0.05, 0.10, 0.05, 0.35],
    [0.05, 0.20, 0.05, 0.00, 0.30, 0.05, 0.35],
])
_BOLD_SHARE = 0.55
_LOUD_THRESH = 0.45


class AdaptiveField(ShapesField):
    NUM_KINDS = 7

    def __init__(self, *args, **kwargs) -> None:
        self._energy = 0.0
        self._energy_ref = 0.0
        super().__init__(*args, **kwargs)
        self._was_active = np.zeros(self._spots, dtype=bool)

    def _set_owner(self) -> None:
        super()._set_owner()
        self._band = np.searchsorted(_BREAKS, self._frac[self._owner]).astype(np.int64)

    def _params(self, kind: np.ndarray) -> tuple:
        rng, n = self._rng, len(kind)
        u = lambda lo, hi: rng.uniform(lo, hi, n)  # noqa: E731
        conds = [kind == ORB, kind == SPIKY, kind == STAR, kind == BLOB, kind == SLASH, kind == WISP]
        elong = np.select(conds, [u(1.0, 1.1), u(1.0, 1.1), u(1.0, 1.25), u(1.0, 1.3), u(2.0, 3.2), u(1.6, 2.6)], u(1.05, 1.35))
        spikes = np.select(conds, [np.zeros(n), rng.integers(11, 19, n), rng.integers(5, 10, n), rng.integers(2, 4, n),
                                   rng.integers(0, 3, n), np.zeros(n)], rng.integers(3, 7, n))
        depth = np.select(conds, [np.zeros(n), u(0.35, 0.6), u(1.3, 2.4), u(0.25, 0.55), u(0.2, 0.5), u(0.05, 0.15)], u(1.0, 1.8))
        expo = np.select([kind == BLOB, kind == SHARD], [np.full(n, 2.0), u(4.0, 5.0)], 3.0)
        return elong.astype(np.float32), spikes.astype(np.int32), depth.astype(np.float32), expo.astype(np.float32)

    def _new_shapes(self) -> None:
        rng, n = self._rng, self._spots
        kind = np.array([rng.choice(7, p=_QUIET[b]) for b in self._band], dtype=np.int64)
        self._kind = kind
        self._elong, self._spikes, self._depth, self._expo = self._params(kind)
        self._angle = rng.uniform(0, np.pi, n).astype(np.float32)
        self._spin = (rng.choice([-1.0, 1.0], n) * rng.uniform(0.008, 0.026, n)).astype(np.float32)

    def reshuffle(self) -> None:
        super().reshuffle()
        self._was_active[:] = False

    def _reroll(self, idx: np.ndarray, boldness: np.ndarray) -> None:
        rng = self._rng
        t = np.clip(boldness, 0.0, 1.0).astype(np.float64)
        t = t * t * (3.0 - 2.0 * t)
        kind = np.empty(len(idx), dtype=np.int64)
        for j, (b, tt) in enumerate(zip(self._band[idx], t)):
            p = (1.0 - tt) * _QUIET[b] + tt * _LOUD[b]
            kind[j] = rng.choice(7, p=p / p.sum())
        e, s, d, x = self._params(kind)
        self._kind[idx] = kind
        self._elong[idx], self._spikes[idx], self._depth[idx], self._expo[idx] = e, s, d, x
        self._angle[idx] = rng.uniform(0, np.pi, len(idx)).astype(np.float32)

    def _shape(self, band_levels: np.ndarray) -> np.ndarray:
        lv = np.asarray(band_levels, dtype=np.float32)
        self._energy = float(lv.mean())
        if float(lv.max()) >= 0.03:
            self._energy_ref = max(self._energy, self._energy_ref * 0.9985)
        else:
            self._energy_ref *= 0.995
        return super()._shape(band_levels)

    def update(self, band_levels: np.ndarray, gain: np.ndarray | None = None) -> np.ndarray:
        self._frame += 1
        self._pos += (self._target - self._pos) * 0.04
        levels = self._levels(band_levels, gain)
        active = levels > 0.05
        rising = np.nonzero(active & ~self._was_active)[0]
        if len(rising):
            frame_t = float(np.clip(self._energy / max(self._energy_ref, 1e-6), 0.0, 1.0))
            self._reroll(rising, np.clip(0.7 * frame_t + 0.3 * levels[rising], 0.0, 1.0))
            levels[~self.kinds_on[self._kind]] = 0.0            # a re-rolled shape may be one that is switched off
        self._was_active = active
        self._suppress_near_loud(levels)
        self._draw(levels)
        return self.buffer

    def _suppress_near_loud(self, levels: np.ndarray) -> None:
        """Quiet detail right next to a loud hit is dimmed, so it shows elsewhere."""
        act = np.nonzero(levels > 0.05)[0]
        loud = levels[act] >= _LOUD_THRESH
        if not (loud.any() and (~loud).any()):
            return
        xy = self.positions()
        centre = np.average(xy[act[loud]], axis=0, weights=levels[act[loud]])
        focus = max(float(np.hypot(self.cluster_w, self.cluster_h)) * 0.3, 1e-3)
        q = act[~loud]
        levels[q] *= np.clip(np.linalg.norm(xy[q] - centre, axis=1) / focus, 0.15, 1.0)

    def _keep(self, area: np.ndarray, active: np.ndarray) -> np.ndarray:
        """Bold and soft shapes each get their own share of the budget."""
        total = self.area_budget * self.cluster_w * self.cluster_h
        bold = np.isin(self._kind[active], BOLD)
        keep = np.zeros(len(active), dtype=bool)
        for mask, cap in ((bold, total * _BOLD_SHARE), (~bold, total * (1 - _BOLD_SHARE))):
            grp = np.nonzero(mask)[0]
            if len(grp):
                keep[grp] = np.cumsum(area[grp]) <= cap
        keep[0] = True
        keep[self.max_active:] = False
        return keep


class AdaptiveEngine(ShapesEngine):
    KEY = "v3"
    LABEL = "V3 Adaptive"
    ELEMENTS = (("orb", "Orbs"), ("spiky", "Spiky orbs"), ("star", "Stars"), ("shard", "Shards"), ("blob", "Blobs"),
                ("wisp", "Wisps"), ("slash", "Slashes"), ("depth", "3D depth"))
    FIELD = AdaptiveField

    def kind_keys(self) -> tuple[str, ...]:
        return KIND_KEYS


__all__ = ["AdaptiveEngine", "scatter"]
