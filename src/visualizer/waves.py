"""Surf: waves of sea foam rolling over the picture like water over a beach.

Each wave is an event, born when a synth, pad, chord or held vocal note begins
(or the note changes, or a long held note "breathes"). It always spans a whole
edge of the screen -- bottom, top, left or right, a different one each time --
and rolls inward: a quick uprush, a pause, then a slower backwash. The front is
never straight and never repeats: every wave draws its own irregular shoreline
(a blend of slow undulations plus a couple of "fingers" of water running ahead),
and that shoreline keeps shifting as the wave travels.

* the leading edge carries stippled, bubbling foam;
* behind it the water sheet inverts what it covers, thinning towards the shore
  side, and the sand it leaves behind stays faintly inverted while it "dries";
* the music sets the size and character: louder notes run further up the beach,
  low sounds make wide coarse foam and bright sounds a fine sparkling line,
  and rich harmonic sounds make a more ragged shoreline.

Between notes there can be no waves at all, and in silence everything fades out.
`compose` inverts whatever is beneath the surf: white -> black, gray flips, and
over black it shows white, so it reads as a depth map across the picture.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# fractions of the 40 Hz - 16 kHz log axis: ~250-700 Hz, 700-2k, 2k-6k, 6k-16k
_LAYER_EDGES = (0.29, 0.47, 0.65, 0.84, 1.0)
EDGES = ("bottom", "top", "left", "right")
MAX_WAVES = 2
_OCTAVE_BINS = 11  # one octave on the 96-bin, 40 Hz - 16 kHz log axis (~11.1 bins)


@dataclass
class Wave:
    edge: str
    reach: float          # furthest inland it runs, as a fraction of the screen (0..1)
    wobble: float         # how ragged the shoreline is
    foam_width: float     # thickness of the foam line
    amp: float
    life: int
    freqs: np.ndarray     # shoreline undulations: cycles across the edge
    phases: np.ndarray
    rates: np.ndarray     # how fast each undulation drifts
    amps: np.ndarray
    fingers: list         # (centre 0..1, width, height, drift) tongues of water running ahead
    foam: np.ndarray      # (h, w) bubble texture, scrolled as the wave moves
    hold: float = 0.15    # share of its life spent at the furthest point
    age: int = 0
    foam_off: float = 0.0


def _smoothstep(lo: float, hi: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def swash(t: float, hold: float) -> float:
    """Where the water is at life fraction t: 0 at the shore, 1 at the furthest
    point. A quick uprush, a pause, then a slower backwash."""
    up_end = 0.26
    down_start = up_end + hold
    if t < up_end:
        return 1.0 - (1.0 - t / up_end) ** 2.4
    if t < down_start:
        return 1.0
    s = (t - down_start) / max(1e-6, 1.0 - down_start)
    return max(0.0, 1.0 - s) ** 1.7


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
        self.softness = softness  # 0 = crisper waterline, 1 = very feathered
        self.rate = rate          # how often waves are born (2 = twice as often)
        self._rng = np.random.default_rng(seed)
        self._edges = [int(round(f * num_bins)) for f in _LAYER_EDGES]
        self._n = len(_LAYER_EDGES) - 1
        self._slow = np.zeros(num_bins, dtype=np.float32)
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
        """False when no wave is on screen and no wet sand is left."""
        return (bool(self._waves) and self._presence > 0.01) or float(self._wet.max()) > 0.01

    @property
    def wave_count(self) -> int:
        return len(self._waves)

    def reshuffle(self) -> None:
        """A new song: let the current waves finish; the next ones start fresh."""
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
        self._along = {"bottom": x, "top": x, "left": y, "right": y}  # position along the edge, 0..1
        self._wet = np.zeros((grid_h, grid_w), dtype=np.float32)
        self._waves.clear()

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
            if fire and self._cool[i] == 0 and self._gap == 0 and len(self._waves) < MAX_WAVES:
                self._spawn(t, int(bins[i]), float(rich[i]))
                self._armed[i] = False
                self._cool[i] = int(240 / max(0.25, self.rate))
                self._hold[i] = 0
                self._interval[i] = int(self._new_interval(1)[0])
                self._anchor[i] = self._where[i]
                self._gap = int(self._rng.integers(300, 600) / max(0.25, self.rate))  # surf comes in sets, not a stream

    def _spawn(self, amp: float, bin_: int, richness: float, edge: str | None = None) -> Wave:
        r = self._rng
        pitch = float(np.clip((bin_ / self.num_bins - 0.29) / 0.71, 0.0, 1.0))  # 0 = low, 1 = high
        if edge is None:                                     # a different edge than last time
            edge = str(r.choice([e for e in EDGES if e != self._last_edge]))
        self._last_edge = edge
        k = int(r.integers(4, 7))
        h, w = self.grid_h, self.grid_w
        foam = r.random((h, w)).astype(np.float32)
        for _ in range(2 if pitch < 0.5 else 1):             # low sounds: coarser bubbles; bright: fine sparkle
            foam = (foam + np.roll(foam, 1, 0) + np.roll(foam, -1, 0) + np.roll(foam, 1, 1) + np.roll(foam, -1, 1)) / 5.0
        foam = (foam - foam.min()) / max(1e-6, float(foam.max() - foam.min()))
        wave = Wave(
            edge=edge,
            reach=float(np.clip(0.28 + 0.6 * amp, 0.3, 0.9) * r.uniform(0.9, 1.1)),
            wobble=float(0.05 + 0.11 * richness) * float(r.uniform(0.8, 1.2)),
            foam_width=float(0.05 - 0.032 * pitch) * float(r.uniform(0.85, 1.2)),
            amp=float(np.clip(0.6 + 0.5 * amp, 0.0, 1.0)),
            life=int(r.integers(300, 460) * (0.85 + 0.3 * amp)),
            freqs=np.sort(r.uniform(0.7, 5.5, k)).astype(np.float32),
            phases=r.uniform(0, 2 * np.pi, k).astype(np.float32),
            rates=r.uniform(-0.03, 0.03, k).astype(np.float32),
            amps=(1.0 / np.arange(1, k + 1) ** 0.7 * r.uniform(0.6, 1.0, k)).astype(np.float32),
            fingers=[(float(r.random()), float(r.uniform(0.04, 0.1)), float(r.uniform(0.05, 0.14)), float(r.uniform(-0.002, 0.002)))
                     for _ in range(int(r.integers(0, 3)))],
            foam=foam,
            hold=float(r.uniform(0.08, 0.28)),
        )
        self._waves.append(wave)
        return wave

    # -- drawing -----------------------------------------------------------------

    def _shoreline(self, wv: Wave, reach_now: float) -> np.ndarray:
        """Distance inland of the water's edge at each position along the screen edge."""
        v = self._along[wv.edge]
        line = np.zeros_like(v, dtype=np.float32)
        for f, ph, rt, a in zip(wv.freqs, wv.phases, wv.rates, wv.amps):
            line = line + a * np.sin(2 * np.pi * f * v + ph + rt * wv.age)
        line = line / max(1e-6, float(wv.amps.sum()))                     # roughly -1..1
        for c, width, height, drift in wv.fingers:
            line = line + (height / max(wv.wobble, 1e-3)) * np.exp(-(((v - (c + drift * wv.age)) / width) ** 2))
        # the ragged part grows with how far up the beach the water is, so the wave
        # starts as a straight line hugging the whole edge and roughens as it runs in
        return reach_now * wv.reach + reach_now * wv.wobble * line

    def _draw(self, wv: Wave, reach_now: float) -> tuple[np.ndarray, np.ndarray]:
        d = self._shoreline(wv, reach_now) - self._dist[wv.edge]   # >0 inside the water
        soft = 0.012 + 0.05 * float(np.clip(self.softness, 0.0, 1.0))
        dry = np.maximum(d, 0.0)
        sheet = 0.5 * _smoothstep(-soft * 0.4, soft, d) * (0.4 + 0.6 * np.exp(-dry / 0.2))
        fw = wv.foam_width
        tex = np.roll(wv.foam, (int(wv.foam_off), int(wv.foam_off * 0.6)), axis=(0, 1))
        bubbles = _smoothstep(0.42, 0.58, tex)                                    # some cells bright, some gaps
        lace = _smoothstep(0.58, 0.72, tex)                                       # sparser, only the brightest specks
        edge = _smoothstep(-fw * 0.7, 0.0, d) * np.exp(-dry / fw)                 # the bright leading line
        trail = _smoothstep(0.0, fw * 0.6, d) * np.exp(-dry / (fw * 3.6))         # lacy foam left behind it
        foam = np.clip(edge * (0.3 + 0.7 * bubbles) + 0.8 * trail * lace, 0.0, 1.0)
        return sheet, foam

    def update(self, band_levels: np.ndarray) -> np.ndarray:
        """Returns the surf layer, shape (grid_h, grid_w), values 0..1."""
        amps, where, bins, rich = self._analyse(band_levels)
        self._detect_onsets(amps, where, bins, rich)

        # presence follows what is audible right now: quiet -> everything fades fast
        now = float(np.clip(float(band_levels.max()) / 0.06, 0.0, 1.0))
        self._presence += (now - self._presence) * (0.15 if now > self._presence else 0.06)

        self._wet *= 0.985                                    # sand slowly dries
        out = np.zeros((self.grid_h, self.grid_w), dtype=np.float32)
        for wv in self._waves:
            wv.age += 1
            wv.foam_off += 0.35
            t = wv.age / wv.life
            fade = min(1.0, wv.age / 40.0) * min(1.0, (wv.life - wv.age) / 60.0)
            alpha = max(0.0, fade) * self._presence * wv.amp * self.strength
            if alpha < 0.005:
                continue
            sheet, foam = self._draw(wv, swash(t, wv.hold))
            np.maximum(self._wet, np.where(sheet > 0.05, 0.3, 0.0).astype(np.float32) * self._presence, out=self._wet)
            s_a, f_a = sheet * alpha, foam * min(1.0, alpha * 1.9)                # foam stays crisp even when the water is faint
            layer = s_a + f_a - s_a * f_a
            out = out + layer - 2.0 * out * layer             # crossing waves invert each other
        self._waves = [wv for wv in self._waves if wv.age < wv.life]
        wet = self._wet * self.strength * 0.5                 # the drying sand is only faintly inverted
        out = out + wet - 2.0 * out * wet
        return np.clip(out, 0.0, 1.0)


def compose(base: np.ndarray, waves: np.ndarray) -> np.ndarray:
    """Waves invert whatever is under them: 1 -> 0, 0 -> 1, 0.3 -> 0.7."""
    return base + waves * (1.0 - 2.0 * base)
