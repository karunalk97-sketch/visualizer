"""V7 -- Field. The song's spectrum, scattered: about 40 frequency bands each own a
spot, clumped at random (a new layout every song), as big as that frequency is loud
-- measured in absolute terms -- and shrinking away when it goes quiet. Each range
draws its own shape: bass ink that melts together, low-mid orbs, vocal rings,
high-mid soft stars, treble sand. Spots drift and turn slowly; overlaps invert."""
from __future__ import annotations

import math

import numpy as np

from ..features import NUM_BANDS, band_edges
from ..imaging import resize_bilinear, smoothstep
from .base import Engine, Inputs

KINDS = ("ink", "orb", "ring", "star", "sand")             # one per range: bass, low mids, vocals, high mids, treble
ELEMENT_OF = {"ink": "ink", "orb": "orbs", "ring": "rings", "star": "stars", "sand": "sand"}
SIZE = {"ink": 0.15, "orb": 0.075, "ring": 0.095, "star": 0.07, "sand": 0.08}   # radius at full loudness, x height
AREA_BUDGET = 0.32


def clumped(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random places that gather in loose clumps, with a few strays -- not evenly spread."""
    centres = rng.uniform(0.12, 0.88, (int(rng.integers(4, 8)), 2))
    spread = rng.uniform(0.05, 0.12, len(centres))
    pts = np.empty((n, 2), dtype=np.float32)
    for i in range(n):
        if rng.random() < 0.75:
            c = int(rng.integers(len(centres)))
            pts[i] = centres[c] + rng.normal(0, spread[c], 2)
        else:
            pts[i] = rng.uniform(0.06, 0.94, 2)
    return np.clip(pts, 0.05, 0.95)


class FieldEngine(Engine):
    KEY = "v7"
    LABEL = "V7 Field"
    ELEMENTS = (("ink", "Ink (bass)"), ("orbs", "Orbs (low mids)"), ("rings", "Rings (vocals)"),
                ("stars", "Stars (high mids)"), ("sand", "Sand (treble)"))
    LOW = 4

    def __init__(self, width: int, height: int, seed: int | None = None) -> None:
        super().__init__(width, height, seed)
        _, rng_of = band_edges()
        self._kind = np.array([KINDS[i] for i in rng_of])
        self.t = 0.0
        self.resize(width, height)
        self.reshuffle()

    def resize(self, width: int, height: int) -> None:
        super().resize(width, height)
        self.h4, self.w4 = max(6, self.H // self.LOW), max(8, self.W // self.LOW)
        self._rows4 = (np.arange(self.h4, dtype=np.float32)[:, None] + 0.5) / self.h4
        self._cols4 = (np.arange(self.w4, dtype=np.float32)[None, :] + 0.5) / self.w4
        self._aspect = self.W / self.H
        self._grains = np.random.default_rng(1234).random((self.H, self.W), dtype=np.float32)

    def reshuffle(self) -> None:
        rng, n = self.rng, NUM_BANDS
        home = clumped(rng, n)
        ink = np.nonzero(self._kind == "ink")[0]                        # the bass ink sits close enough to merge
        home[ink] = np.clip(rng.uniform(0.3, 0.7, 2) + rng.normal(0, 0.13, (len(ink), 2)), 0.1, 0.9)
        self._home = home
        self._drift_amp = rng.uniform(0.01, 0.035, (n, 2)).astype(np.float32)
        self._drift_rate = rng.uniform(0.05, 0.14, (n, 2)).astype(np.float32)
        self._drift_phase = rng.uniform(0, 2 * math.pi, (n, 2)).astype(np.float32)
        self._angle = rng.uniform(0, 2 * math.pi, n).astype(np.float32)
        self._spin = (rng.choice([-1.0, 1.0], n) * rng.uniform(0.15, 0.45, n)).astype(np.float32)
        self._points = rng.choice([5, 6, 7], n)
        self._stretch = rng.uniform(1.0, 1.25, n).astype(np.float32)

    def update(self, inp: Inputs, dt: float) -> np.ndarray:
        f = inp.features
        dt = float(min(max(dt, 1e-3), 0.1))
        self.t += dt
        self._angle += self._spin * dt
        H, W = self.H, self.W
        on = np.array([self.on(ELEMENT_OF[k]) for k in self._kind])
        level = np.clip(np.asarray(f.bands, dtype=np.float32), 0.0, 1.5) * on
        pop = np.where(self._kind == "ink", 0.25 * f.kick_env,
                       np.where((self._kind == "star") | (self._kind == "sand"), 0.2 * f.hit_env, 0.0))
        size = np.array([SIZE[k] for k in self._kind], dtype=np.float32)
        radius = size * np.power(level, 0.8) * (1.0 + pop)
        area = float(np.sum(math.pi * (radius * H) ** 2))
        if area > AREA_BUDGET * W * H:
            radius *= math.sqrt(AREA_BUDGET * W * H / area)
        pos = self._home + self._drift_amp * np.sin(self.t * self._drift_rate * 2 * math.pi + self._drift_phase)

        img = self._draw_ink(pos, radius)
        if img is None:
            img = self.blank()
        for i in np.nonzero((self._kind != "ink") & (radius * H > 1.2))[0]:
            self._draw_spot(img, i, pos[i, 0] * W, pos[i, 1] * H, float(radius[i] * H))
        np.clip(img, 0.0, 1.0, out=img)
        return img

    def _draw_ink(self, pos: np.ndarray, radius: np.ndarray) -> np.ndarray | None:
        idx = np.nonzero((self._kind == "ink") & (radius > 0.004))[0]
        if not len(idx):
            return None
        ax = self._aspect
        X, Y = self._cols4 * ax, self._rows4
        fld = np.zeros((self.h4, self.w4), dtype=np.float32)
        for i in idx:
            r = float(radius[i])
            fld += (r * r) / ((X - pos[i, 0] * ax) ** 2 + (Y - pos[i, 1]) ** 2 + 1e-5)
        return resize_bilinear(smoothstep(0.9, 1.1, fld), self.H, self.W)

    def _draw_spot(self, img: np.ndarray, i: int, cx: float, cy: float, r: float) -> None:
        H, W = self.H, self.W
        kind = self._kind[i]
        reach = r * float(self._stretch[i]) + 2
        x0, x1 = max(0, int(cx - reach)), min(W, int(cx + reach) + 2)
        y0, y1 = max(0, int(cy - reach)), min(H, int(cy + reach) + 2)
        if x1 <= x0 or y1 <= y0:
            return
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
        a = float(self._angle[i])
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        d = np.sqrt((u / float(self._stretch[i])) ** 2 + v * v)
        if kind == "orb":
            shape = smoothstep(r + 0.7, r - 0.7, d)
        elif kind == "ring":
            thick = max(1.2, 0.34 * r)
            shape = smoothstep(r + 0.7, r - 0.7, d) * smoothstep(r - thick - 0.7, r - thick + 0.7, d)
        elif kind == "star":
            edge = r * (0.62 + 0.38 * np.cos(int(self._points[i]) * np.arctan2(v, u) / 2.0) ** 4)
            shape = smoothstep(edge + 0.7, edge - 0.7, d)
        else:
            density = np.clip(1.0 - d / max(r, 1e-3), 0.0, 1.0) * 0.75
            shape = (self._grains[y0:y1, x0:x1] < density).astype(np.float32)
        box = img[y0:y1, x0:x1]
        img[y0:y1, x0:x1] = box + shape - (2.0 * self.invert) * box * shape
