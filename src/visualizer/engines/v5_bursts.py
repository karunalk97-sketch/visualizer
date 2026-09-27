"""V5 -- Bursts. Ink that follows the voice, and a radial burst wherever a hit lands:
a kick flares a grey halo around a white core and the hardest send a ring out and
pulse the picture; a snare or hat shoots sunburst rays and pops a spinning star.
Strong hits fire several bursts at once; the sharp parts leave a glow trail."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..imaging import resize_bilinear, smoothstep, value_noise
from .base import Engine, Inputs

MAX_BURSTS = 14


def _scatter(rng, n, lo=0.1, hi=0.9) -> np.ndarray:
    pts = np.empty((n, 2), dtype=np.float32)
    pts[0] = rng.uniform(lo, hi, 2)
    for i in range(1, n):
        cand = rng.uniform(lo, hi, (10, 2)).astype(np.float32)
        d = ((cand[:, None, :] - pts[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        pts[i] = cand[int(np.argmax(d))]
    return pts


@dataclass
class _Burst:
    x: float
    y: float
    kind: str
    parts: frozenset
    strength: float
    size: float
    life: float
    angle: float = 0.0
    spin: float = 0.0
    age: float = 0.0
    phase: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


class BurstsEngine(Engine):
    KEY = "v5"
    LABEL = "V5 Bursts"
    ELEMENTS = (("ink", "Ink"), ("halos", "Halos"), ("cores", "Cores"), ("rings", "Rings"), ("spokes", "Sunbursts"),
                ("stars", "Stars"), ("trails", "Glow trails"), ("pulse", "Pulse"), ("grain", "Grain"))
    LOW = 4
    _KICK = (("halo", "halos"), ("core", "cores"), ("ring", "rings"))
    _SNARE = (("spokes", "spokes"), ("star", "stars"), ("grain", "grain"))

    def __init__(self, width: int, height: int, seed: int | None = None) -> None:
        super().__init__(width, height, seed)
        self.t = 0.0
        self.resize(width, height)
        self.reshuffle()

    def resize(self, width: int, height: int) -> None:
        super().resize(width, height)
        self.h4, self.w4 = max(6, self.H // self.LOW), max(8, self.W // self.LOW)
        self._rows = np.arange(self.H, dtype=np.float32)[:, None]
        self._cols = np.arange(self.W, dtype=np.float32)[None, :]
        self._rows4 = (np.arange(self.h4, dtype=np.float32)[:, None] + 0.5) / self.h4
        self._cols4 = (np.arange(self.w4, dtype=np.float32)[None, :] + 0.5) / self.w4
        self._aspect = self.W / self.H
        self._trail = np.zeros((self.H, self.W), dtype=np.float32)

    def reshuffle(self) -> None:
        r = self.rng
        self._kick_w = dict(zip(("halo", "core", "ring"), r.dirichlet([1.5] * 3)))
        self._snare_w = dict(zip(("spokes", "star", "grain"), r.dirichlet([1.5] * 3)))
        self._spokes = int(r.choice([8, 10, 12, 16, 20]))
        self._points = int(r.choice([4, 5, 5, 6, 8]))
        self._trail_s = float(r.uniform(0.18, 0.4))
        self._ink_n = int(r.integers(2, 5))
        self._ink_size = float(r.uniform(0.85, 1.25))
        self._anchors = _scatter(r, 16)
        self._lattice = r.random((16, 16)).astype(np.float32)
        self._ink_phase = r.uniform(0, 2 * math.pi, (self._ink_n, 4)).astype(np.float32)
        self._ink_speed = r.uniform(0.04, 0.11, (self._ink_n, 2)).astype(np.float32)
        centre = r.uniform(0.38, 0.62, 2)
        self._ink_home = np.clip(centre + r.normal(0, 0.14, (self._ink_n, 2)), 0.2, 0.8).astype(np.float32)
        self._bursts: list[_Burst] = []
        self._ripples: list[_Burst] = []
        self._pulse = self._grain = self._ink = self._lift = 0.0
        self._recent: dict[int, float] = {}
        self._quiet = 0.0
        self._trail[:] = 0.0

    def new_section(self, section: float) -> None:
        self._anchors = _scatter(np.random.default_rng(self.rng.integers(0, 2**31 - 1)), 16)

    # -- per frame ---------------------------------------------------------------

    def update(self, inp: Inputs, dt: float) -> np.ndarray:
        f = inp.features
        dt = float(min(max(dt, 1e-3), 0.1))
        self.t += dt
        calm, stark = 1.0 - f.section, f.section
        coverage = f.level ** 0.8
        self._quiet = self._quiet + dt if f.silent else 0.0
        live = self._quiet < 0.3
        target = 0.0 if not live or not self.on("ink") else min(1.4, (0.25 + 0.85 * min(1.0, f.sustain)) * (0.6 + 0.5 * f.level))
        tau = 0.6 if target > self._ink else (0.5 if not live else 1.6)
        self._ink += (target - self._ink) * (1.0 - math.exp(-dt / tau))
        self._lift += ((f.pitch - 0.5) - self._lift) * (1.0 - math.exp(-dt / 0.6))
        hiss = 0.0
        if live:
            if f.kick > 0.03:
                self._on_kick(min(1.2, f.kick), calm, stark, coverage)
            if f.hit > 0.03:
                self._on_hit(min(1.2, f.hit), f.brightness, calm, stark, coverage)
            hiss = f.high * f.noisiness * 0.25 if self.on("grain") else 0.0
        self._grain = max(self._grain * math.exp(-dt / 0.12), hiss * self._snare_w["grain"])
        self._pulse *= math.exp(-dt / 0.22)

        ink = self._draw_ink(f)
        glow, sharp = self._draw_bursts(dt)
        if self.on("trails"):
            np.maximum(self._trail * math.exp(-dt / (self._trail_s * (0.75 + 0.6 * calm))), sharp, out=self._trail)
            if self._quiet > 1.0:
                self._trail[:] = 0.0
            sharp = self._trail
        events = sharp if glow is None else self.xor(glow, sharp)
        img = self.xor(ink, events) if ink is not None else events.copy()
        if self._grain >= 0.02:
            img = self.xor(img, (self.rng.random((self.H, self.W), dtype=np.float32) < self._grain * 0.035).astype(np.float32))
        img = self._displace(img)
        self._age(dt)
        np.clip(img, 0.0, 1.0, out=img)
        return img

    def _draw_ink(self, f) -> np.ndarray | None:
        amount = self._ink
        if amount < 0.02:
            return None
        ax, t = self._aspect, self.t
        X, Y = self._cols4 * ax, self._rows4
        warp = 0.16 * (0.6 + 0.8 * f.sustain)
        wu = value_noise(self._lattice, X * 2.2 + t * 0.07, Y * 2.2 - t * 0.05) - 0.5
        wv = value_noise(self._lattice, X * 2.2 + 7.3 - t * 0.06, Y * 2.2 + 3.1 + t * 0.04) - 0.5
        Xw, Yw = X + wu * warp * ax, Y + wv * warp
        fld = np.zeros((self.h4, self.w4), dtype=np.float32)
        for k in range(self._ink_n):
            ph, sp, home = self._ink_phase[k], self._ink_speed[k], self._ink_home[k]
            cx = (home[0] + 0.2 * math.sin(t * sp[0] * 2 * math.pi + ph[0]) + 0.06 * math.sin(t * 0.7 + ph[2])) * ax
            cy = home[1] - 0.22 * self._lift + 0.18 * math.sin(t * sp[1] * 2 * math.pi + ph[1]) + 0.05 * math.cos(t * 0.5 + ph[3])
            r = 0.16 * self._ink_size * math.sqrt(min(amount, 1.4)) * (0.55 + 0.45 * f.level) * (1 + 0.12 * self._pulse)
            fld += (r * r) / ((Xw - cx) ** 2 + (Yw - cy) ** 2 + 1e-4)
        core = smoothstep(0.85, 1.15, fld)
        return resize_bilinear(core + 0.4 * smoothstep(0.35, 0.85, fld) * (1.0 - core), self.H, self.W)

    # -- bursts ------------------------------------------------------------------

    def _anchor(self) -> tuple[float, float]:
        free = [i for i in range(len(self._anchors)) if self.t - self._recent.get(i, -9.0) > 0.5]
        i = int(self.rng.choice(free)) if free else int(self.rng.integers(len(self._anchors)))
        self._recent[i] = self.t
        return float(self._anchors[i][0] + self.rng.normal(0, 0.03)), float(self._anchors[i][1] + self.rng.normal(0, 0.03))

    def _choose(self, table, weights: dict, lean: dict, n: int) -> set:
        names = [part for part, element in table if self.on(element)]
        if not names:
            return set()
        p = np.array([weights[k] * lean[k] for k in names]) + 1e-6
        pick = self.rng.choice(len(names), size=min(n, len(names)), replace=False, p=p / p.sum())
        return {names[i] for i in np.atleast_1d(pick)}

    def _spawn(self, **kw) -> _Burst | None:
        if len(self._bursts) >= MAX_BURSTS:
            return None
        b = _Burst(**kw)
        b.phase = self.rng.uniform(0, 2 * math.pi, 4).astype(np.float32)
        self._bursts.append(b)
        return b

    def _on_kick(self, k: float, calm: float, stark: float, coverage: float) -> None:
        lean = {"halo": 0.8 + 0.6 * calm, "core": 1.0, "ring": 0.5 + 1.0 * stark}
        for n in range(1 + (k > 0.55) + (k > 0.85)):
            parts = self._choose(self._KICK, self._kick_w, lean, 1 + (k > 0.4) + (k > 0.75))
            if self.on("halos"):
                parts.add("halo")
            if not parts:
                break
            x, y = self._anchor()
            size = (0.12 + 0.22 * min(k, 1.0)) * (0.6 + 0.7 * coverage) * (1.0 if n == 0 else 0.7)
            b = self._spawn(x=x, y=y, kind="kick", parts=frozenset(parts), strength=min(1.0, 0.5 + 0.6 * k), size=size,
                            life=0.22 + 0.25 * calm)
            if b is not None and "ring" in parts and len(self._ripples) < 3:
                self._ripples.append(b)
        if k > 0.7 and self.on("pulse"):
            self._pulse = max(self._pulse, min(1.0, k) * (0.5 + 0.5 * coverage))

    def _on_hit(self, h: float, bright: float, calm: float, stark: float, coverage: float) -> None:
        lean = {"spokes": 0.6 + 0.9 * stark, "star": (0.5 + 0.8 * stark) * (0.5 + bright), "grain": 0.6 + 0.8 * calm}
        for n in range(1 + (h > 0.6) + (h > 0.9)):
            parts = self._choose(self._SNARE, self._snare_w, lean, 1 + (h > 0.5))
            if "grain" in parts:
                self._grain = max(self._grain, min(1.0, h) * (0.6 + 0.6 * coverage))
                parts.discard("grain")
            if not parts:
                continue
            x, y = self._anchor()
            self._spawn(x=x, y=y, kind="snare", parts=frozenset(parts), strength=min(1.0, 0.5 + 0.6 * h),
                        size=(0.08 + 0.14 * min(h, 1.0)) * (0.6 + 0.7 * coverage) * (1.0 if n == 0 else 0.75),
                        life=0.16 + 0.14 * calm, angle=float(self.rng.uniform(0, 2 * math.pi)),
                        spin=float(self.rng.choice([-1, 1]) * self.rng.uniform(0.6, 1.6)))

    def _draw_bursts(self, dt: float) -> tuple[np.ndarray | None, np.ndarray]:
        H, W = self.H, self.W
        sharp = np.zeros((H, W), dtype=np.float32)
        if not self._bursts:
            return None, sharp
        ax = self._aspect
        X, Y = self._cols4 * ax, self._rows4
        soft = np.zeros((self.h4, self.w4), dtype=np.float32)
        cores = np.zeros((self.h4, self.w4), dtype=np.float32)
        any_soft = any_core = False
        for b in self._bursts:
            rise = 1.0 - math.exp(-b.age / 0.035)
            fade = math.exp(-b.age / b.life)
            amp = rise * fade * b.strength
            if amp < 0.03:
                continue
            if b.kind == "kick":
                d = np.sqrt((X - b.x * ax) ** 2 + (Y - b.y) ** 2)
                grow = 0.6 + 0.4 * rise + 0.35 * (1.0 - math.exp(-b.age / 0.5))
                if "halo" in b.parts:
                    soft = self.xor(soft, 0.6 * amp * np.exp(-((d / (b.size * 0.85 * grow)) ** 2)))
                    any_soft = True
                if "core" in b.parts:
                    cores = self.xor(cores, np.clip((1.0 - d / (b.size * 0.5 * grow)) * 6.0 * amp, 0.0, 1.0))
                    any_core = True
                if "ring" in b.parts:
                    R = b.size * (0.3 + 1.5 * (1.0 - math.exp(-b.age / 0.25)))
                    soft = self.xor(soft, np.clip(2.0 * amp * np.exp(-(((d - R) / (b.size * 0.09)) ** 2)), 0.0, 1.0))
                    any_soft = True
            else:
                b.angle += b.spin * dt
                reach = b.size * H * (0.35 + 0.65 * rise) * (0.6 + 0.4 * fade)
                if "spokes" in b.parts:
                    self._draw_spokes(sharp, b, reach * 1.6, fade)
                if "star" in b.parts:
                    self._draw_star(sharp, b, reach * 0.7)
        if any_core:
            sharp = self.xor(resize_bilinear(cores, H, W), sharp)
        return (resize_bilinear(soft, H, W) if any_soft else None), sharp

    def _draw_spokes(self, img: np.ndarray, b: _Burst, reach: float, fade: float) -> None:
        H, W = self.H, self.W
        cx, cy = b.x * W, b.y * H
        inner = reach * (0.18 + 0.6 * (1.0 - fade))
        if reach - inner < 2:
            return
        s = np.linspace(inner, reach, int(reach - inner) + 2, dtype=np.float32)
        for k in range(self._spokes):
            a = b.phase[2] + 2 * math.pi * k / self._spokes + 0.08 * math.sin(k * 1.7 + b.phase[0])
            length = 0.7 + 0.3 * math.sin(k * 2.3 + b.phase[1])
            xs = np.round(cx + math.cos(a) * s * length).astype(np.int32)
            ys = np.round(cy + math.sin(a) * s * length).astype(np.int32)
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            img[ys[ok], xs[ok]] = 1.0

    def _draw_star(self, img: np.ndarray, b: _Burst, radius: float) -> None:
        H, W = self.H, self.W
        if radius < 1.5:
            return
        cx, cy = b.x * W, b.y * H
        x0, x1 = max(0, int(cx - radius)), min(W, int(cx + radius) + 1)
        y0, y1 = max(0, int(cy - radius)), min(H, int(cy + radius) + 1)
        if x1 <= x0 or y1 <= y0:
            return
        dx = np.arange(x0, x1, dtype=np.float32)[None, :] - cx
        dy = np.arange(y0, y1, dtype=np.float32)[:, None] - cy
        th = np.arctan2(dy, dx) - b.angle
        edge = radius * (0.38 + 0.62 * np.abs(np.cos(self._points * th / 2.0)) ** 3)
        img[y0:y1, x0:x1][np.sqrt(dx * dx + dy * dy) <= edge] = 1.0

    def _displace(self, img: np.ndarray) -> np.ndarray:
        active = [r for r in self._ripples if r.age < 0.9]
        z = 0.03 * self._pulse
        if not active and z < 0.002:
            return img
        H, W = self.H, self.W
        h2, w2 = (H + 1) // 2, (W + 1) // 2
        X2 = np.arange(w2, dtype=np.float32)[None, :] * 2.0
        Y2 = np.arange(h2, dtype=np.float32)[:, None] * 2.0
        dx = np.zeros((h2, w2), dtype=np.float32)
        dy = np.zeros((h2, w2), dtype=np.float32)
        for r in active:
            ox, oy = X2 - r.x * W, Y2 - r.y * H
            d = np.sqrt(ox * ox + oy * oy) + 1e-3
            u = (d - r.size * H * (0.3 + 1.5 * (1.0 - math.exp(-r.age / 0.25)))) / (r.size * H * 0.35)
            near = np.abs(u) < 3.0
            push = np.zeros_like(d)
            push[near] = r.strength * max(0.0, 1.0 - r.age / 0.9) * 0.03 * H * np.exp(-u[near] ** 2) * np.sin(math.pi * u[near])
            dx += ox / d * push
            dy += oy / d * push
        if z >= 0.002:
            dx += (X2 - W / 2) * z
            dy += (Y2 - H / 2) * z
        dx = np.repeat(np.repeat(dx, 2, axis=0), 2, axis=1)[:H, :W]
        dy = np.repeat(np.repeat(dy, 2, axis=0), 2, axis=1)[:H, :W]
        sx = np.clip(np.rint(self._cols - dx), 0, W - 1).astype(np.intp)
        sy = np.clip(np.rint(self._rows - dy), 0, H - 1).astype(np.intp)
        return img[sy, sx]

    def _age(self, dt: float) -> None:
        live = {id(b) for b in self._bursts}
        for b in self._bursts:
            b.age += dt
        for r in self._ripples:
            if id(r) not in live:
                r.age += dt
        self._bursts = [b for b in self._bursts if math.exp(-b.age / b.life) * b.strength >= 0.03]
        self._ripples = [r for r in self._ripples if r.age < 0.9]
