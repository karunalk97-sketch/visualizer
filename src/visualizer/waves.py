"""Cascading waves driven by sustained, tonal sound -- synths, pads, harmonies,
held vocal notes.

Percussion and bass hits come and go too fast to count: each frequency band is
compared with its own recent average and only the energy that has *stayed* (the
lower of the two) drives a wave. Four wave layers cover the mid and high range,
and each one wears the pitch of what is playing there: higher notes make
tighter bands, louder ones make faster, thicker, stronger bands. Bands are
domain-warped so they snake across the screen instead of ruling straight
lines, and every layer drifts and turns slowly.

Waves are not drawn as white on top of the picture. `compose` inverts whatever
is underneath them, so a wave over solid white cuts a black line, over gray it
flips the gray, and over black it shows as white -- a depth map across
everything else. When the music goes quiet the waves fade out completely.
"""
from __future__ import annotations

import numpy as np

# fractions of the 40 Hz - 16 kHz log axis: ~250-700 Hz, 700-2k, 2k-6k, 6k-16k
_LAYER_EDGES = (0.29, 0.47, 0.65, 0.84, 1.0)


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class WaveField:
    def __init__(
        self,
        num_bins: int,
        cluster_w: int,
        cluster_h: int,
        strength: float = 0.6,
        seed: int | None = None,
    ) -> None:
        self.num_bins = num_bins
        self.strength = strength
        self._rng = np.random.default_rng(seed)
        self._edges = [int(round(f * num_bins)) for f in _LAYER_EDGES]
        self._n = len(_LAYER_EDGES) - 1
        self._slow = np.zeros(num_bins, dtype=np.float32)
        self._peak_ref = 0.0
        self._seg_ref = np.zeros(len(_LAYER_EDGES) - 1, dtype=np.float32)
        self._amp = np.zeros(self._n, dtype=np.float32)
        self._wavelen = np.full(self._n, 0.15, dtype=np.float32)
        self._phase = self._rng.uniform(0, 2 * np.pi, self._n).astype(np.float32)
        self._new_layout()
        self.resize(cluster_w, cluster_h)

    def _new_layout(self) -> None:
        r, n = self._rng, self._n
        self._angle = r.uniform(0, np.pi, n).astype(np.float32)
        self._turn = r.uniform(-0.0018, 0.0018, n).astype(np.float32)
        self._dir = r.choice([-1.0, 1.0], n).astype(np.float32)
        self._warp_phase = r.uniform(0, 2 * np.pi, n).astype(np.float32)
        self._warp_len = r.uniform(0.6, 1.2, n).astype(np.float32)
        self._warp_drift = r.uniform(0.01, 0.03, n).astype(np.float32)
        self._harm = r.uniform(0.1, 0.5, n).astype(np.float32)
        # true peak of each layer's profile, so every wave line can reach full strength
        p = np.linspace(0, 2 * np.pi, 720)[None, :]
        self._prof_peak = (np.sin(p) + self._harm[:, None] * np.sin(2 * p + 0.7)).max(axis=1).astype(np.float32)

    @property
    def active(self) -> bool:
        """False when every layer has faded out (silence, or nothing sustained)."""
        return float(self._amp.max()) >= 0.01

    def reshuffle(self) -> None:
        """A new song gets new directions and shapes for its waves."""
        self._new_layout()

    def resize(self, cluster_w: int, cluster_h: int) -> None:
        self.cluster_w, self.cluster_h = cluster_w, cluster_h
        aspect = cluster_w / max(1, cluster_h)
        self._x = np.linspace(0.0, aspect, cluster_w, dtype=np.float32)[None, :]
        self._y = np.linspace(0.0, 1.0, cluster_h, dtype=np.float32)[:, None]

    # -- analysis ----------------------------------------------------------------

    def _layer_targets(self, band_levels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per layer: how strongly it should show (0..1) and where in its range
        the strongest sustained pitch sits (0 = low edge, 1 = high edge)."""
        self._slow = 0.97 * self._slow + 0.03 * band_levels
        sustained = np.minimum(band_levels, self._slow)  # only what has stayed
        peak = float(band_levels.max())
        self._peak_ref = max(peak, self._peak_ref * 0.995)
        ref = max(self._peak_ref, 1e-6)
        loud = min(1.0, ref / 0.2)  # silence stays silent
        amps = np.zeros(self._n, dtype=np.float32)
        where = np.full(self._n, 0.5, dtype=np.float32)
        if self._peak_ref < 0.03:
            return amps, where
        for i in range(self._n):
            lo, hi = self._edges[i], self._edges[i + 1]
            # each layer is judged against its own loudness, so a pad still shows
            # through loud drums and bass that dominate the overall level
            self._seg_ref[i] = max(float(band_levels[lo:hi].max()), self._seg_ref[i] * 0.995)
            seg_ref = max(float(self._seg_ref[i]), 1e-6)
            seg = sustained[lo:hi]
            top = float(seg.max())
            present = min(1.0, seg_ref / 0.06)  # a near-empty band makes no waves
            amps[i] = np.clip((top / seg_ref - 0.45) / 0.35, 0.0, 1.0) * present * loud
            where[i] = float(seg.argmax()) / max(1, len(seg) - 1)
        return amps, where

    # -- per frame ---------------------------------------------------------------

    def update(self, band_levels: np.ndarray) -> np.ndarray:
        """Returns the wave layer, shape (cluster_h, cluster_w), values 0..1."""
        target, where = self._layer_targets(band_levels)
        rate = np.where(target > self._amp, 0.10, 0.04)  # come in quickly, leave gently
        self._amp += (target - self._amp) * rate
        # pitch -> spacing: higher notes in a layer's range make tighter bands
        goal = (0.20 - 0.11 * where) * np.linspace(1.0, 0.65, self._n).astype(np.float32)
        self._wavelen += (goal - self._wavelen) * 0.03

        out = np.zeros((self.cluster_h, self.cluster_w), dtype=np.float32)
        if float(self._amp.max()) < 0.01:
            return out

        # several layers at once would pile up into a busy mesh: share the intensity
        crowd = 1.0 / (1.0 + 0.3 * max(0, int((self._amp > 0.1).sum()) - 1))

        self._angle += self._turn
        self._phase += self._dir * (0.05 + 0.10 * self._amp)
        self._warp_phase += self._warp_drift
        for i in range(self._n):
            a = float(self._amp[i])
            if a < 0.01:
                continue
            ca, sa = np.cos(self._angle[i]), np.sin(self._angle[i])
            u = self._x * ca + self._y * sa
            v = self._y * ca - self._x * sa
            wl = float(self._wavelen[i])
            snake = 0.9 * wl * np.sin(2 * np.pi * v / float(self._warp_len[i]) + self._warp_phase[i])
            phase = 2 * np.pi * (u + snake) / wl - self._phase[i]
            h = float(self._harm[i])
            prof = (np.sin(phase) + h * np.sin(2 * phase + 0.7)) / float(self._prof_peak[i])
            line = _smoothstep(0.90 - 0.04 * a, 0.985, 0.5 + 0.5 * prof)
            layer = line * a * self.strength * crowd
            out = out + layer - 2.0 * out * layer  # crossing waves invert each other too
        return np.clip(out, 0.0, 1.0)


def compose(base: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """Waves invert whatever is under them: 1 -> 0, 0 -> 1, 0.3 -> 0.7."""
    return base + waves * (1.0 - 2.0 * base)
