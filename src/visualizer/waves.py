"""The foam line: one smooth line of foam that sweeps across the picture.

Think of the crest of a wave -- just the thick foam, nothing behind it. Each wave
is an event, born when a synth, pad, chord or held vocal note begins (or the note
changes, or a long held note "breathes"). It enters from one edge of the screen (a
different edge each time), spans the whole edge, and travels across to the far
side and off it. That is all: no wash of water behind it, no wet sand, no
retreat.

The line is shaped by the music, and it stays shaped by it while it travels:

* the sustained tonal energy in the mid and high range -- what synths, pads and
  harmonies are doing, not the drums -- is smoothed into a gentle profile along the
  line: where that energy is strong the line bulges forward *and* thickens, where
  it is weak the line thins and trails;
* smoothing is heavy on purpose, so the line is a soft flowing curve with no
  jagged edges, and its edges fade gaussian-soft;
* louder notes cross faster, low sounds make a thicker line, bright sounds a
  thinner one.

The line carries a soft bubbly foam texture. `compose` inverts whatever is beneath
it: over black it shows as bright foam, over white it cuts black, over gray it
flips the gray.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# fractions of the 40 Hz - 16 kHz log axis: ~250-700 Hz, 700-2k, 2k-6k, 6k-16k
_LAYER_EDGES = (0.29, 0.47, 0.65, 0.84, 1.0)
EDGES = ("bottom", "top", "left", "right")
MAX_WAVES = 2
PROFILE_POINTS = 48    # samples of the music's shape along the line
_OCTAVE_BINS = 11      # one octave on the 96-bin, 40 Hz - 16 kHz log axis (~11.1 bins)
_MARGIN = 0.34         # the line starts and ends this far outside the screen (room for its bulge and thickness)


@dataclass
class Wave:
    edge: str
    width: float          # thickness of the line where the music is strong (fraction of the screen)
    bulge: float          # how far the music can push the line forward
    amp: float
    life: int             # frames to cross the screen
    flip: bool            # which end of the line the low frequencies sit at
    profile: np.ndarray   # (PROFILE_POINTS,) smoothed sustained energy along the line, 0..1
    foam: np.ndarray      # (h, w) bubble texture, scrolled as the wave moves
    age: int = 0
    foam_off: float = 0.0


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth_profile(values: np.ndarray, points: int = PROFILE_POINTS, passes: int = 4) -> np.ndarray:
    """Resample a spectrum to `points` samples and blur it heavily: a gentle,
    flowing shape with no sharp peaks, normalised to 0..1."""
    v = np.asarray(values, dtype=np.float32)
    if v.max() <= 1e-9:
        return np.zeros(points, dtype=np.float32)
    out = np.interp(np.linspace(0, len(v) - 1, points), np.arange(len(v)), v).astype(np.float32)
    kernel = np.array([1, 4, 6, 4, 1], dtype=np.float32) / 16.0
    for _ in range(passes):
        out = np.convolve(np.pad(out, 2, mode="edge"), kernel, mode="valid")
    lo, hi = float(out.min()), float(out.max())                       # stretch to the full range so the shape shows
    return ((out - lo) / max(hi - lo, 1e-9)).astype(np.float32) if hi - lo > 1e-6 * hi else np.full(points, 0.5, np.float32)


class WaveField:
    def __init__(
        self,
        num_bins: int,
        grid_w: int,
        grid_h: int,
        strength: float = 0.6,
        softness: float = 0.6,
        rate: float = 1.0,
        seed: int | None = None,
    ) -> None:
        self.num_bins = num_bins
        self.strength = strength
        self.softness = softness  # 0 = crisper edges, 1 = very feathered
        self.rate = rate          # how often waves are born (2 = twice as often)
        self._rng = np.random.default_rng(seed)
        self._edges = [int(round(f * num_bins)) for f in _LAYER_EDGES]
        self._n = len(_LAYER_EDGES) - 1
        self._slow = np.zeros(num_bins, dtype=np.float32)
        self._sustained = np.zeros(num_bins, dtype=np.float32)
        self._peak_ref = 0.0
        self._seg_ref = np.zeros(self._n, dtype=np.float32)
        self._presence = 0.0
        self._waves: list[Wave] = []
        self._last_edge = ""
        # per-layer onset detection state
        self._armed = np.ones(self._n, dtype=bool)
        self._cool = np.zeros(self._n, dtype=np.int32)
        self._hold = np.zeros(self._n, dtype=np.int32)
        self._interval = self._new_interval(self._n)
        self._where = np.full(self._n, 0.5, dtype=np.float32)
        self._anchor = np.full(self._n, 0.5, dtype=np.float32)
        self._gap = 0
        self.resize(grid_w, grid_h)

    def _new_interval(self, n: int) -> np.ndarray:
        """Frames a held note waits before it may 'breathe' out another wave."""
        return (self._rng.integers(700, 1300, n) / max(0.25, self.rate)).astype(np.int32)

    # -- public state ------------------------------------------------------------

    @property
    def active(self) -> bool:
        """False when no wave is on screen."""
        return bool(self._waves) and self._presence > 0.01

    @property
    def wave_count(self) -> int:
        return len(self._waves)

    def reshuffle(self) -> None:
        """A new song: let the current wave finish; the next ones start fresh."""
        self._armed[:] = True
        self._cool[:] = 0

    def resize(self, grid_w: int, grid_h: int) -> None:
        self.grid_w, self.grid_h = grid_w, grid_h
        y = np.linspace(0.0, 1.0, grid_h, dtype=np.float32)[:, None]
        x = np.linspace(0.0, 1.0, grid_w, dtype=np.float32)[None, :]
        self._dist = {  # distance inland from each edge, 0 at the edge, 1 at the far side
            "bottom": np.broadcast_to(1.0 - y, (grid_h, grid_w)),
            "top": np.broadcast_to(y, (grid_h, grid_w)),
            "left": np.broadcast_to(x, (grid_h, grid_w)),
            "right": np.broadcast_to(1.0 - x, (grid_h, grid_w)),
        }
        # position along the edge (0..1): x for the horizontal edges, y for the vertical ones
        self._along = {"bottom": x, "top": x, "left": y, "right": y}
        self._waves.clear()

    # -- analysis ----------------------------------------------------------------

    def _analyse(self, band_levels: np.ndarray):
        """Per layer: strength of sustained energy (0..1), where in its range the
        strongest sustained pitch sits (0..1), its bin, and its harmonic richness."""
        self._slow = 0.97 * self._slow + 0.03 * band_levels
        sustained = np.minimum(band_levels, self._slow)  # only what has stayed: synths, pads, harmonies
        self._sustained = sustained
        self._peak_ref = max(float(band_levels.max()), self._peak_ref * 0.995)
        loud = min(1.0, self._peak_ref / 0.2)
        amps = np.zeros(self._n, dtype=np.float32)
        where = np.full(self._n, 0.5, dtype=np.float32)
        bins = np.zeros(self._n, dtype=np.int32)
        rich = np.zeros(self._n, dtype=np.float32)
        if self._peak_ref < 0.03:
            return amps, where, bins, rich
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
            k = int(seg.argmax())
            bins[i] = lo + k
            where[i] = k / max(1, len(seg) - 1)
            up = bins[i] + _OCTAVE_BINS
            rich[i] = np.clip(sustained[up] / max(float(sustained[bins[i]]), 1e-6), 0.0, 1.0) if up < self.num_bins else 0.3
        return amps, where, bins, rich

    def _music_profile(self) -> np.ndarray:
        """The smoothed shape of the sustained mid/high spectrum: the music, seen
        as a gentle curve, 0..1."""
        return smooth_profile(self._sustained[self._edges[0]:])

    # -- births ------------------------------------------------------------------

    def _detect_onsets(self, amps, where, bins, rich) -> None:
        self._gap = max(0, self._gap - 1)
        self._cool = np.maximum(0, self._cool - 1)
        self._where += (where - self._where) * 0.1  # the loudest bin jitters; follow it smoothly
        for i in range(self._n):
            t = float(amps[i])
            if t < 0.12:
                self._armed[i] = True
                self._hold[i] = 0
                continue
            if t < 0.3:
                continue
            self._hold[i] += 1
            fire = (
                self._armed[i]                                            # a note/chord just started
                or abs(self._where[i] - self._anchor[i]) > 0.3            # the pitch moved: new note
                or self._hold[i] > self._interval[i]                      # a long held note breathes
            )
            self._anchor[i] += (self._where[i] - self._anchor[i]) * 0.02
            if fire and self._cool[i] == 0 and self._gap == 0 and len(self._waves) < MAX_WAVES:
                self._spawn(t, int(bins[i]), float(rich[i]))
                self._armed[i] = False
                self._cool[i] = int(240 / max(0.25, self.rate))
                self._hold[i] = 0
                self._interval[i] = int(self._new_interval(1)[0])
                self._anchor[i] = self._where[i]
                self._gap = int(self._rng.integers(420, 800) / max(0.25, self.rate))  # occasional, not a stream

    def _spawn(self, amp: float, bin_: int, richness: float, edge: str | None = None) -> Wave:
        r = self._rng
        pitch = float(np.clip((bin_ / self.num_bins - 0.29) / 0.71, 0.0, 1.0))  # 0 = low, 1 = high
        if edge is None:                                     # a different edge than last time
            edge = str(r.choice([e for e in EDGES if e != self._last_edge]))
        self._last_edge = edge
        h, w = self.grid_h, self.grid_w
        foam = r.random((h, w)).astype(np.float32)
        for _ in range(3 if pitch < 0.5 else 2):             # soft bubbles, coarser for low sounds
            foam = (foam + np.roll(foam, 1, 0) + np.roll(foam, -1, 0) + np.roll(foam, 1, 1) + np.roll(foam, -1, 1)) / 5.0
        foam = (foam - foam.min()) / max(1e-6, float(foam.max() - foam.min()))
        wave = Wave(
            edge=edge,
            width=float((0.085 - 0.045 * pitch) * r.uniform(0.9, 1.15)),
            bulge=float(0.26 + 0.1 * richness) * float(r.uniform(0.85, 1.15)),
            amp=float(np.clip(0.65 + 0.5 * amp, 0.0, 1.0)),
            life=int(r.integers(380, 560) * (1.25 - 0.4 * amp)),      # louder notes cross faster
            flip=bool(r.random() < 0.5),
            profile=self._music_profile(),
            foam=foam,
        )
        self._waves.append(wave)
        return wave

    # -- drawing -----------------------------------------------------------------

    def _position(self, wv: Wave) -> float:
        """Progress of the line across the screen: starts outside the source edge,
        ends outside the far edge. Steady, with the gentlest ease at both ends."""
        t = wv.age / wv.life
        e = 0.75 * t + 0.25 * t * t * (3.0 - 2.0 * t)
        return -_MARGIN + (1.0 + 2.0 * _MARGIN) * e

    def _line(self, wv: Wave) -> tuple[np.ndarray, np.ndarray]:
        """(centre, thickness) of the line at each position along the screen edge."""
        v = self._along[wv.edge]
        n = v.shape[1] if v.shape[0] == 1 else v.shape[0]
        prof = wv.profile[::-1] if wv.flip else wv.profile
        p = np.interp(np.linspace(0.0, 1.0, n), np.linspace(0.0, 1.0, len(prof)), prof).astype(np.float32)
        p = p.reshape(v.shape)
        centre = self._position(wv) + (p - float(p.mean())) * wv.bulge     # the music pushes the line forward
        thick = np.sqrt((wv.width * (0.3 + 1.3 * p)) ** 2 + 0.03 ** 2)      # ...and thickens it; never a hair-thin line
        return centre, thick

    def _draw(self, wv: Wave) -> np.ndarray:
        centre, thick = self._line(wv)
        dd = self._dist[wv.edge] - centre                    # distance from the line's centre, across it
        k = 2.4 * (1.0 - 0.6 * float(np.clip(self.softness, 0.0, 1.0)))   # gaussian falloff: soft edges, never sharp
        band = np.exp(-((dd / np.maximum(thick, 1e-3)) ** 2) * k)
        tex = np.roll(wv.foam, (int(wv.foam_off), int(wv.foam_off * 0.6)), axis=(0, 1))
        bubbles = _smoothstep(0.3, 0.7, tex)                 # a soft bubbly texture, never hard specks
        return (band * (0.6 + 0.4 * bubbles)).astype(np.float32)

    def update(self, band_levels: np.ndarray) -> np.ndarray:
        """Returns the foam layer, shape (grid_h, grid_w), values 0..1."""
        amps, where, bins, rich = self._analyse(band_levels)
        self._detect_onsets(amps, where, bins, rich)

        # presence follows what is audible right now: quiet -> everything fades fast
        now = float(np.clip(float(band_levels.max()) / 0.06, 0.0, 1.0))
        self._presence += (now - self._presence) * (0.15 if now > self._presence else 0.06)

        out = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        live = self._music_profile()
        for wv in self._waves:
            wv.age += 1
            wv.foam_off += 0.3
            if float(live.max()) > 0.0:                      # keep listening: the line follows the music as it travels
                wv.profile += (live - wv.profile) * 0.02
            alpha = min(1.0, wv.age / 25.0) * min(1.0, (wv.life - wv.age) / 25.0) * self._presence * wv.amp * self.strength
            if alpha < 0.005:
                continue
            layer = np.clip(self._draw(wv) * min(1.0, alpha * 2.4), 0.0, 1.0)
            out = out + layer - 2.0 * out * layer            # crossing lines invert each other
        self._waves = [wv for wv in self._waves if wv.age < wv.life]
        return np.clip(out, 0.0, 1.0)


def compose(base: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """Waves invert whatever is under them: 1 -> 0, 0 -> 1, 0.3 -> 0.7."""
    return base + waves * (1.0 - 2.0 * base)
