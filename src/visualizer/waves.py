"""Soft ribbon waves, born when a synth, pad, chord or held vocal note begins.

There is no constant wave field. Each ribbon is an event: it appears when
sustained tonal energy starts (or the note/chord changes, or a long held note
"breathes" again), drifts across the screen, and fades. Between notes there can
be no waves at all, and in silence everything fades out.

The frequencies picked up shape each ribbon:

* low, warm sounds  -> fat, slow, spindle-shaped swells (thick in the middle);
* bright sounds     -> thin, tighter ribbons;
* rich / harmonic   -> a wobble along the ribbon and, mid-range, a dumbbell
                       shape (thicker towards the ends).

Edges are gaussian-soft, never sharp. Ribbons are computed on a small grid and
smoothly upscaled by the caller. `compose` inverts whatever is beneath a wave,
so a ribbon over solid white cuts black, over gray it flips the gray, and over
black it shows white.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# fractions of the 40 Hz - 16 kHz log axis: ~250-700 Hz, 700-2k, 2k-6k, 6k-16k
_LAYER_EDGES = (0.29, 0.47, 0.65, 0.84, 1.0)
MAX_RIBBONS = 3
_OCTAVE_BINS = 11  # one octave on the 96-bin, 40 Hz - 16 kHz log axis (~11.1 bins)


@dataclass
class Ribbon:
    kind: str            # "spindle" (thick middle), "dumbbell" (thick ends), "flat"
    x0: float            # start position, in screen-height units
    y0: float
    dx: float            # direction of travel (unit vector); the ribbon lies across it
    dy: float
    speed: float         # height-units per frame
    length: float
    width: float         # thickness at its thickest
    bend: float
    wobble: float
    wob_len: float
    phase: float
    amp: float
    life: int
    age: int = 0


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


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
        self.softness = softness  # 0 = crisper ribbon edges, 1 = very feathered
        self.rate = rate          # how often waves are born (2 = twice as often)
        self._rng = np.random.default_rng(seed)
        self._edges = [int(round(f * num_bins)) for f in _LAYER_EDGES]
        self._n = len(_LAYER_EDGES) - 1
        self._slow = np.zeros(num_bins, dtype=np.float32)
        self._peak_ref = 0.0
        self._seg_ref = np.zeros(self._n, dtype=np.float32)
        self._presence = 0.0
        self._ribbons: list[Ribbon] = []
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
        """Frames a held note waits before it may 'breathe' out another ribbon."""
        return (self._rng.integers(700, 1300, n) / max(0.25, self.rate)).astype(np.int32)

    # -- public state ------------------------------------------------------------

    @property
    def active(self) -> bool:
        """False when no ribbon is on screen (silence, or nothing tonal playing)."""
        return bool(self._ribbons) and self._presence > 0.01

    @property
    def ribbon_count(self) -> int:
        return len(self._ribbons)

    def reshuffle(self) -> None:
        """A new song: let the current ribbons finish; new ones get new directions."""
        self._armed[:] = True
        self._cool[:] = 0

    def resize(self, grid_w: int, grid_h: int) -> None:
        self.grid_w, self.grid_h = grid_w, grid_h
        aspect = grid_w / max(1, grid_h)
        self.aspect = aspect
        self._x = np.linspace(0.0, aspect, grid_w, dtype=np.float32)[None, :]
        self._y = np.linspace(0.0, 1.0, grid_h, dtype=np.float32)[:, None]

    # -- analysis ----------------------------------------------------------------

    def _analyse(self, band_levels: np.ndarray):
        """Per layer: strength of sustained energy (0..1), where in its range the
        strongest sustained pitch sits (0..1), its bin, and its harmonic richness."""
        self._slow = 0.97 * self._slow + 0.03 * band_levels
        sustained = np.minimum(band_levels, self._slow)  # only what has stayed
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
            if fire and self._cool[i] == 0 and self._gap == 0 and len(self._ribbons) < MAX_RIBBONS:
                self._spawn(i, t, int(bins[i]), float(rich[i]))
                self._armed[i] = False
                self._cool[i] = int(240 / max(0.25, self.rate))
                self._hold[i] = 0
                self._interval[i] = int(self._new_interval(1)[0])
                self._anchor[i] = self._where[i]
                self._gap = int(self._rng.integers(420, 800) / max(0.25, self.rate))  # waves are occasional, not a stream

    def _spawn(self, layer: int, amp: float, bin_: int, richness: float) -> None:
        r = self._rng
        pitch = float(np.clip((bin_ / self.num_bins - 0.29) / 0.71, 0.0, 1.0))  # 0 = low, 1 = high
        kind = "flat" if pitch > 0.7 else ("dumbbell" if (richness > 0.55 and pitch > 0.25) else "spindle")
        width = (0.075 - 0.058 * pitch) * r.uniform(0.85, 1.2)
        length = self.aspect * (1.05 - 0.5 * pitch) * r.uniform(0.85, 1.15)
        theta = r.uniform(0, 2 * np.pi)
        dx, dy = float(np.cos(theta)), float(np.sin(theta))
        life = int(r.integers(300, 480))
        speed = (0.9 + 0.5 * amp) * (1.3 - 0.6 * pitch) / life * r.uniform(0.9, 1.1)  # crosses ~1 screen
        cx, cy = 0.5 * self.aspect + r.uniform(-0.25, 0.25), 0.5 + r.uniform(-0.2, 0.2)
        travel = speed * life
        self._ribbons.append(Ribbon(
            kind=kind,
            x0=cx - dx * travel / 2, y0=cy - dy * travel / 2, dx=dx, dy=dy, speed=speed,
            length=length, width=width,
            bend=float(r.uniform(-0.35, 0.35)),
            wobble=float(width * richness * r.uniform(0.6, 1.4)),
            wob_len=float(r.uniform(0.18, 0.5) * (1.0 - 0.4 * pitch)),
            phase=float(r.uniform(0, 2 * np.pi)),
            amp=float(np.clip(0.55 + 0.6 * amp, 0.0, 1.0)),
            life=life,
        ))

    # -- drawing -----------------------------------------------------------------

    def _draw(self, rb: Ribbon, k: float) -> np.ndarray:
        cx = rb.x0 + rb.dx * rb.speed * rb.age
        cy = rb.y0 + rb.dy * rb.speed * rb.age
        rx, ry = self._x - cx, self._y - cy
        along = -rx * rb.dy + ry * rb.dx            # position along the ribbon's long axis
        across = rx * rb.dx + ry * rb.dy            # distance across it
        across = across - rb.bend * along * along   # gentle arc
        across = across + rb.wobble * np.sin(2 * np.pi * along / rb.wob_len + rb.phase + rb.age * 0.025)
        s = np.clip(along / rb.length + 0.5, 0.0, 1.0)
        c = np.cos(np.pi * (s - 0.5))               # 1 in the middle, 0 at the tips
        if rb.kind == "spindle":
            taper = np.maximum(c, 0.0) ** 0.8       # thick in the middle
        elif rb.kind == "dumbbell":
            taper = 0.35 + 0.65 * (1.0 - c) ** 1.2  # thick towards the ends
        else:
            taper = np.minimum(1.0, c * 3.0)        # even thin ribbon
        d = np.abs(across) / (rb.width * taper + 1e-3)
        band = np.exp(-(d * d) * k)                 # gaussian: soft edges, never sharp
        tips = _smoothstep(0.0, 0.14, s) * _smoothstep(1.0, 0.86, s)
        inside = ((along > -rb.length / 2) & (along < rb.length / 2)).astype(np.float32)
        return band * tips * inside

    def update(self, band_levels: np.ndarray) -> np.ndarray:
        """Returns the wave layer, shape (grid_h, grid_w), values 0..1."""
        amps, where, bins, rich = self._analyse(band_levels)
        self._detect_onsets(amps, where, bins, rich)

        # presence follows what is audible right now: quiet -> everything fades fast
        now = float(np.clip(float(band_levels.max()) / 0.06, 0.0, 1.0))
        self._presence += (now - self._presence) * (0.15 if now > self._presence else 0.06)

        out = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        if not self._ribbons:
            return out
        k = 3.4 * (1.0 - 0.75 * float(np.clip(self.softness, 0.0, 1.0)))  # softness -> gaussian falloff
        for rb in self._ribbons:
            rb.age += 1
            fade = min(1.0, rb.age / 70.0) * min(1.0, (rb.life - rb.age) / 110.0)
            alpha = max(0.0, fade) * self._presence * rb.amp * self.strength
            if alpha < 0.005:
                continue
            layer = self._draw(rb, k) * alpha
            out = out + layer - 2.0 * out * layer  # crossing ribbons invert each other
        self._ribbons = [rb for rb in self._ribbons if rb.age < rb.life]
        return np.clip(out, 0.0, 1.0)


def compose(base: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """Waves invert whatever is under them: 1 -> 0, 0 -> 1, 0.3 -> 0.7."""
    return base + waves * (1.0 - 2.0 * base)
