"""V6 -- Ink waves. Soft ink that follows the voice; every hit starts a hollow, wobbly
ink shape (ink with no center) where it lands, which spreads across the whole
screen, thinning and dissolving as it goes -- thick for a hard kick, thin for a
snare -- inverting whatever it passes over. Nothing sharp."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..imaging import resize_bilinear, smoothstep, value_noise
from .base import Engine, Inputs

MAX_WAVES = 10
SLOWER = 1.6                     # these waves used to cross the screen too quickly


def _scatter(rng, n, lo=0.12, hi=0.88) -> np.ndarray:
    pts = np.empty((n, 2), dtype=np.float32)
    pts[0] = rng.uniform(lo, hi, 2)
    for i in range(1, n):
        cand = rng.uniform(lo, hi, (10, 2)).astype(np.float32)
        d = ((cand[:, None, :] - pts[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        pts[i] = cand[int(np.argmax(d))]
    return pts


@dataclass
class _Wave:
    x: float
    y: float
    thick: float
    tau: float
    reach: float
    age: float = 0.0

    def radius(self) -> float:
        return self.reach * (1.0 - math.exp(-self.age / self.tau))

    def progress(self) -> float:
        return self.radius() / self.reach

    def band(self) -> float:
        return self.thick * max(0.0, 1.0 - self.progress()) ** 0.6

    def strength(self) -> float:
        return max(0.0, 1.0 - self.progress() / 0.9) ** 0.8


class InkWavesEngine(Engine):
    KEY = "v6"
    LABEL = "V6 Ink waves"
    ELEMENTS = (("ink", "Ink"), ("waves", "Ink waves"), ("pulse", "Pulse"), ("grain", "Grain"))
    LOW = 4

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

    def reshuffle(self) -> None:
        r = self.rng
        self._wobble = float(r.uniform(0.14, 0.24))
        self._speed = float(r.uniform(0.8, 1.25))
        self._thickness = float(r.uniform(0.8, 1.3))
        self._ink_n = int(r.integers(2, 5))
        self._ink_size = float(r.uniform(0.85, 1.25))
        self._anchors = _scatter(r, 16)
        self._lattice = r.random((16, 16)).astype(np.float32)
        self._ink_phase = r.uniform(0, 2 * math.pi, (self._ink_n, 4)).astype(np.float32)
        self._ink_speed = r.uniform(0.04, 0.11, (self._ink_n, 2)).astype(np.float32)
        centre = r.uniform(0.38, 0.62, 2)
        self._ink_home = np.clip(centre + r.normal(0, 0.14, (self._ink_n, 2)), 0.2, 0.8).astype(np.float32)
        self._waves: list[_Wave] = []
        self._pulse = self._grain = self._ink = self._lift = 0.0
        self._recent: dict[int, float] = {}
        self._quiet = 0.0

    def new_section(self, section: float) -> None:
        self._anchors = _scatter(np.random.default_rng(self.rng.integers(0, 2**31 - 1)), 16)

    def update(self, inp: Inputs, dt: float) -> np.ndarray:
        f = inp.features
        dt = float(min(max(dt, 1e-3), 0.1))
        self.t += dt
        calm = 1.0 - f.section
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
                self._on_hit(min(1.2, f.kick), "kick", calm, coverage)
            if f.hit > 0.03:
                self._on_hit(min(1.2, f.hit), "snare", calm, coverage)
            hiss = f.high * f.noisiness * 0.12 if self.on("grain") else 0.0
        else:
            self._waves.clear()
        self._grain = max(self._grain * math.exp(-dt / 0.12), hiss)
        self._pulse *= math.exp(-dt / 0.25)

        low = self._draw_ink(f)
        waves = self._draw_waves()
        if waves is not None:
            low = waves if low is None else self.xor(low, waves)
        img = resize_bilinear(low, self.H, self.W) if low is not None else self.blank()
        if self._grain >= 0.02:
            img = self.xor(img, (self.rng.random((self.H, self.W), dtype=np.float32) < self._grain * 0.02).astype(np.float32))
        img = self._pulse_zoom(img)
        for w in self._waves:
            w.age += dt
        self._waves = [w for w in self._waves if w.strength() > 0.03]
        np.clip(img, 0.0, 1.0, out=img)
        return img

    def _warped(self, scale: float, amount: float, offset: float):
        ax, t = self._aspect, self.t
        X, Y = self._cols4 * ax, self._rows4
        wu = value_noise(self._lattice, X * scale + t * 0.07 + offset, Y * scale - t * 0.05) - 0.5
        wv = value_noise(self._lattice, X * scale + 7.3 - t * 0.06, Y * scale + 3.1 + t * 0.04 + offset) - 0.5
        return X + wu * amount * ax, Y + wv * amount

    def _draw_ink(self, f) -> np.ndarray | None:
        amount = self._ink
        if amount < 0.02:
            return None
        ax, t = self._aspect, self.t
        Xw, Yw = self._warped(2.2, 0.16 * (0.6 + 0.8 * f.sustain), 0.0)
        fld = np.zeros((self.h4, self.w4), dtype=np.float32)
        for k in range(self._ink_n):
            ph, sp, home = self._ink_phase[k], self._ink_speed[k], self._ink_home[k]
            cx = (home[0] + 0.2 * math.sin(t * sp[0] * 2 * math.pi + ph[0]) + 0.06 * math.sin(t * 0.7 + ph[2])) * ax
            cy = home[1] - 0.22 * self._lift + 0.18 * math.sin(t * sp[1] * 2 * math.pi + ph[1]) + 0.05 * math.cos(t * 0.5 + ph[3])
            r = 0.16 * self._ink_size * math.sqrt(min(amount, 1.4)) * (0.55 + 0.45 * f.level) * (1 + 0.12 * self._pulse)
            fld += (r * r) / ((Xw - cx) ** 2 + (Yw - cy) ** 2 + 1e-4)
        core = smoothstep(0.85, 1.15, fld)
        return core + 0.35 * smoothstep(0.4, 0.85, fld) * (1.0 - core)

    def _on_hit(self, strength: float, kind: str, calm: float, coverage: float) -> None:
        s = min(strength, 1.0)
        if self.on("waves"):
            for n in range(1 + (s > 0.85) + (kind == "kick" and s > 0.95)):
                if len(self._waves) >= MAX_WAVES:
                    break
                free = [i for i in range(16) if self.t - self._recent.get(i, -9.0) > 0.5]
                i = int(self.rng.choice(free)) if free else int(self.rng.integers(16))
                self._recent[i] = self.t
                x, y = float(self._anchors[i][0]), float(self._anchors[i][1])
                reach = max(math.hypot((cx - x) * self._aspect, cy - y) for cx in (0.0, 1.0) for cy in (0.0, 1.0)) + 0.15
                if kind == "kick":
                    thick, tau = (0.035 + 0.13 * s) * (0.6 + 0.6 * coverage), (0.5 + 0.25 * calm) / self._speed
                else:
                    thick, tau = (0.015 + 0.05 * s) * (0.6 + 0.6 * coverage), (0.32 + 0.15 * calm) / self._speed
                self._waves.append(_Wave(x, y, thick * self._thickness * (1.0 if n == 0 else 0.7), tau * SLOWER, reach))
        if kind == "kick" and s > 0.75 and self.on("pulse"):
            self._pulse = max(self._pulse, s * (0.5 + 0.5 * coverage))

    def _draw_waves(self) -> np.ndarray | None:
        if not self._waves:
            return None
        Xw, Yw = self._warped(1.3, self._wobble, 11.0)
        out = np.zeros((self.h4, self.w4), dtype=np.float32)
        for w in self._waves:
            fade = w.strength()
            if fade < 0.03:
                continue
            outer, inner = w.radius(), w.radius() - w.band()
            d = np.sqrt((Xw - w.x * self._aspect) ** 2 + (Yw - w.y) ** 2)
            soft = 0.012 * (1.0 + 3.0 * w.progress())
            ring = smoothstep(inner - soft, inner + soft, d) * (1.0 - smoothstep(outer - soft, outer + soft, d))
            out = self.xor(out, ring * fade)
        return out

    def _pulse_zoom(self, img: np.ndarray) -> np.ndarray:
        z = 0.025 * self._pulse
        if z < 0.002:
            return img
        H, W = self.H, self.W
        sx = np.clip(np.rint(W / 2 + (self._cols - W / 2) * (1 - z)), 0, W - 1).astype(np.intp)
        sy = np.clip(np.rint(H / 2 + (self._rows - H / 2) * (1 - z)), 0, H - 1).astype(np.intp)
        return img[sy, sx]
