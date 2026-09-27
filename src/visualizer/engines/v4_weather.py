"""V4 -- Weather. Layers the music builds: a perspective ridgeline terrain drawn from
the actual waveform, cymatic sand, a silk ribbon that rides the melody, and ink --
one of them leads each song (the section can hand the lead to another) -- with kick
blooms that dissolve into sand, ripples and a pulse on the biggest hits, and grain,
hairlines and small shards for bright, noisy sounds."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..features import Features
from ..imaging import resize_bilinear, smoothstep, value_noise
from .base import Engine, Inputs

SUSTAIN = ("terrain", "ribbon", "ink", "sand")
MAX_BLOOMS, MAX_RIPPLES, MAX_SHARDS, MAX_LINES = 5, 2, 10, 12


def _scatter(rng: np.random.Generator, n: int, lo: float = 0.1, hi: float = 0.9) -> np.ndarray:
    pts = np.empty((n, 2), dtype=np.float32)
    pts[0] = rng.uniform(lo, hi, 2)
    for i in range(1, n):
        cand = rng.uniform(lo, hi, (10, 2)).astype(np.float32)
        d = ((cand[:, None, :] - pts[None, :i, :]) ** 2).sum(axis=2).min(axis=1)
        pts[i] = cand[int(np.argmax(d))]
    return pts


def _line_top(y: np.ndarray) -> np.ndarray:
    left = np.concatenate([y[:1], y[:-1]])
    right = np.concatenate([y[1:], y[-1:]])
    return np.minimum(y, np.minimum(left, right) + 1)


def _weights(rng, names, alpha, floor=None) -> dict:
    w = {n: float(v) for n, v in zip(names, rng.dirichlet([alpha] * len(names)))}
    for n, v in (floor or {}).items():
        w[n] = max(w[n], v)
    s = sum(w.values())
    return {n: v / s for n, v in w.items()}


@dataclass
class _Event:
    x: float
    y: float
    strength: float
    age: float = 0.0
    life: float = 1.0
    size: float = 0.0
    angle: float = 0.0
    spin: float = 0.0
    phase: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))


class WeatherEngine(Engine):
    KEY = "v4"
    LABEL = "V4 Weather"
    ELEMENTS = (("terrain", "Terrain"), ("ribbon", "Ribbon"), ("ink", "Ink"), ("sand", "Sand"), ("blooms", "Blooms"),
                ("ripples", "Ripples"), ("grain", "Grain"), ("hairlines", "Hairlines"), ("shards", "Shards"))
    LOW = 4

    def __init__(self, width: int, height: int, seed: int | None = None) -> None:
        super().__init__(width, height, seed)
        self.t = 0.0
        self.resize(width, height)
        self.reshuffle()

    # -- setup -------------------------------------------------------------------

    def resize(self, width: int, height: int) -> None:
        super().resize(width, height)
        H, W = self.H, self.W
        self.h4, self.w4 = max(6, H // self.LOW), max(8, W // self.LOW)
        self._rows = np.arange(H, dtype=np.float32)[:, None]
        self._cols = np.arange(W, dtype=np.float32)[None, :]
        self._rows4 = (np.arange(self.h4, dtype=np.float32)[:, None] + 0.5) / self.h4
        self._cols4 = (np.arange(self.w4, dtype=np.float32)[None, :] + 0.5) / self.w4
        self._aspect = W / H
        self._ridge_hist = np.zeros((64, W), dtype=np.float32)
        self._stipple = np.random.default_rng(1234).random((H, W), dtype=np.float32) * 0.98 + 0.02
        if hasattr(self, "_mix_w"):
            self._layout_terrain()

    def reshuffle(self) -> None:
        r = self.rng
        self._mix_w = {
            "kick": _weights(r, ("bloom", "ripple", "pulse"), 1.0, {"bloom": 0.3}),
            "bright": _weights(r, ("grain", "lines", "shards", "texture"), 0.9),
            "sustain": _weights(r, SUSTAIN, 0.8),
        }
        self._ridges = int(r.integers(14, 30))
        self._horizon = float(r.uniform(0.12, 0.4))
        self._span = float(r.uniform(0.35, 0.9))
        self._center = float(r.uniform(0.4, 0.6))
        self._ridge_amp = float(r.uniform(1.5, 3.2))
        self._ridge_speed = float(r.uniform(7.0, 16.0))
        self._thick = int(r.choice([1, 1, 2]))
        self._persp = float(r.uniform(1.5, 4.0))
        self._line_angle = float(r.choice([0.0, 0.0, math.pi / 2, math.pi / 6, -math.pi / 6, math.pi / 3, -math.pi / 3]))
        self._blob_n = int(r.integers(2, 5))
        self._blob_size = float(r.uniform(0.8, 1.3))
        self._strands = int(r.integers(5, 10))
        self._wave_freq = float(r.uniform(1.2, 3.5))
        self._grain_size = int(r.choice([1, 1, 2]))
        self._anchors = _scatter(r, 14)
        self._lattice = r.random((16, 16)).astype(np.float32)
        self._grain_lattice = r.random((12, 12)).astype(np.float32)
        self._sand_scale = 4.0
        self._ribbon_y = self.H * 0.5
        self._blob_phase = r.uniform(0, 2 * math.pi, (self._blob_n, 4)).astype(np.float32)
        self._blob_speed = r.uniform(0.04, 0.12, (self._blob_n, 2)).astype(np.float32)
        centre = r.uniform(0.35, 0.65, 2)
        self._blob_home = np.clip(centre + r.normal(0, 0.14, (self._blob_n, 2)), 0.2, 0.8).astype(np.float32)
        self._wave_phase = r.uniform(0, 2 * math.pi, 2).astype(np.float32)
        self._ridge_hist[:] = 0.0
        self._ridge_phase = 0.0
        self._layout_terrain()
        self._blooms: list[_Event] = []
        self._ripples: list[_Event] = []
        self._shards: list[_Event] = []
        self._lines: list[_Event] = []
        self._pulse = self._grain = self._texture = 0.0
        self._presence = {k: 0.0 for k in SUSTAIN}
        self._section = 0.5
        self._layers = self.lead_layers(0.5)
        self._recent: dict[int, float] = {}
        self._quiet = 0.0

    def set_elements(self, keys) -> None:
        before = set(self.enabled)
        super().set_elements(keys)
        if self.enabled != before and hasattr(self, "_mix_w"):
            self._layers = self.lead_layers(self._section)

    def _layout_terrain(self) -> None:
        xs = np.linspace(0.0, 1.0, self.W, dtype=np.float32)
        dist = np.abs(xs - self._center) / max(self._span / 2, 1e-3)
        self._edge = np.clip(1.0 - (dist / 0.8) ** 4, 0.0, 1.0).astype(np.float32)
        self._line_cols = (dist <= 1.0).astype(np.float32)

    def new_section(self, section: float) -> None:
        self._section = section
        self._anchors = _scatter(np.random.default_rng(self.rng.integers(0, 2**31 - 1)), 14)
        self._layers = self.lead_layers(section)

    def lead_layers(self, section: float) -> tuple[str | None, str | None]:
        """The song's leading sustained layer and at most one supporting one, among
        those switched on. The terrain only ever leads; terrain and ribbon never share."""
        calm, stark = 1.0 - section, section
        lean = {"terrain": 0.7 + 0.6 * stark, "ribbon": 0.7 + 0.6 * calm, "ink": 0.5 + 0.9 * calm, "sand": 0.5 + 0.7 * calm}
        w = self._mix_w["sustain"]
        eff = sorted((k for k in SUSTAIN if self.on(k)), key=lambda k: w[k] * lean[k], reverse=True)
        if not eff:
            return None, None
        lead = eff[0]
        second = next((k for k in eff[1:] if k != "terrain" and {lead, k} != {"terrain", "ribbon"}), None)
        if second is not None and w[second] * lean[second] < 0.15 * w[lead] * lean[lead]:
            second = None
        return lead, second

    # -- per frame ---------------------------------------------------------------

    def update(self, inp: Inputs, dt: float) -> np.ndarray:
        f = inp.features
        dt = float(min(max(dt, 1e-3), 0.1))
        self.t += dt
        self._section = f.section
        calm, stark = 1.0 - f.section, f.section
        coverage = f.level ** 0.9
        self._quiet = self._quiet + dt if f.silent else 0.0
        live = self._quiet < 0.3
        self._sustained(f, dt)
        if live:
            if f.kick > 0:
                self._on_kick(f, calm, stark, coverage)
            if f.hit > 0:
                self._on_hit(f, calm, stark, coverage)
        hiss = f.high * f.noisiness * (0.5 + 0.5 * calm) if live and self.on("grain") else 0.0
        self._grain = max(self._grain * math.exp(-dt / 0.12), 0.4 * hiss * self._mix_w["bright"]["grain"])
        self._texture *= math.exp(-dt / 0.16)
        self._pulse *= math.exp(-dt / 0.22)

        img = self._draw_back(f)
        front = self._draw_front(f)
        if front is not None:
            img = self.xor(img, front)
        img = self._draw_sharp(img, dt)
        img = self._draw_grain(img)
        img = self._displace(img)
        self._age(dt)
        np.clip(img, 0.0, 1.0, out=img)
        return img

    def _sustained(self, f: Features, dt: float) -> None:
        lead, second = self._layers
        silent = self._quiet > 0.3
        for k in SUSTAIN:
            share = 1.0 if k == lead else 0.7 if k == second else 0.0
            target = 0.0 if silent else min(1.0, (0.25 + 0.9 * f.sustain) * share * (0.6 + 0.5 * f.level))
            attack, release = (1.0, 2.5) if k == "terrain" else (0.8, 2.0)
            if silent:
                release = 0.6
            tau = attack if target > self._presence[k] else release
            self._presence[k] += (target - self._presence[k]) * (1.0 - math.exp(-dt / tau))
        self._ridge_phase += dt * self._ridge_speed * (0.7 + 0.6 * f.level)
        while self._ridge_phase >= 1.0:
            self._ridge_phase -= 1.0
            self._ridge_hist[1:] = self._ridge_hist[:-1]
            self._ridge_hist[0] = self._ridge_from(f)

    def _ridge_from(self, f: Features) -> np.ndarray:
        x = f.waveform.astype(np.float32)
        k = 120
        if len(x) < k + 16:
            return np.zeros(self.W, dtype=np.float32)
        c = np.cumsum(np.concatenate([[0.0], x]))
        lp = ((c[k:] - c[:-k]) / k).astype(np.float32)
        half = len(lp) // 2
        cross = np.nonzero((lp[:half - 1] <= 0) & (lp[1:half] > 0))[0]
        start = int(cross[0]) if len(cross) else 0
        seg = lp[start:start + half]
        if len(seg) < 8:
            return np.zeros(self.W, dtype=np.float32)
        cols = np.interp(np.linspace(0, len(seg) - 1, self.W), np.arange(len(seg)), seg).astype(np.float32)
        smooth = int(round(4 + 10 * (1.0 - f.brightness)))
        kern = np.hanning(smooth + 2)[1:-1]
        cols = np.convolve(cols, kern / kern.sum(), mode="same").astype(np.float32) * 7.0
        cols = np.where(cols > 0, cols, cols * 0.3)
        return np.clip(cols, -0.6, 2.0) * self._edge

    # -- back: sand + terrain ------------------------------------------------------

    def _draw_back(self, f: Features) -> np.ndarray:
        H, W = self.H, self.W
        sand_p = self._presence["sand"]
        img = self._draw_sand(f, sand_p) if sand_p > 0.01 else np.zeros((H, W), dtype=np.float32)
        p = self._presence["terrain"]
        if p < 0.01:
            return img
        n = self._ridges
        horizon = self._horizon * H
        depth = (np.arange(n, dtype=np.float32) + self._ridge_phase - 1.0) / n
        g = (1.0 - depth) / (1.0 + self._persp * np.maximum(depth, 0.0))
        base = horizon + (H * 1.02 - horizon) * g
        spacing = np.abs(np.gradient(base)) + 1e-3
        amp = spacing * self._ridge_amp * (0.35 + 0.65 * p) * (1.0 + 0.25 * self._pulse)
        narrow = 0.6 + 0.4 * np.clip(g, 0, 1)
        xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
        topi = np.empty((n, W), dtype=np.int32)
        cols = np.empty((n, W), dtype=bool)
        for i in range(n):
            src = (xs - 0.5) / narrow[i] + 0.5
            ridge = np.interp(src, xs, self._ridge_hist[i], left=0.0, right=0.0)
            cols[i] = np.interp(src, xs, self._line_cols, left=0.0, right=0.0) > 0.5
            topi[i] = np.floor(base[i] - amp[i] * ridge).astype(np.int32)
        skyline = np.full(W, H + 10, dtype=np.int32)
        lines = np.zeros((H, W), dtype=np.float32)
        visible = (spacing >= 4.5) & (np.arange(n) < n * min(1.0, 1.4 * p - 0.05))
        for i in range(n):
            y = np.where(cols[i], topi[i], H + 10)
            if not visible[i]:
                continue
            lo = _line_top(y)
            hi = np.where(cols[i], np.minimum(y + self._thick - 1, skyline - 1), -1)
            r0, r1 = max(0, int(lo.min())), min(H - 1, int(hi.max()))
            if r1 >= r0:
                rows = np.arange(r0, r1 + 1, dtype=np.int32)[:, None]
                lines[r0:r1 + 1][(rows >= lo[None, :]) & (rows <= hi[None, :])] = 1.0
            np.minimum(skyline, y, out=skyline)
        if sand_p > 0.01:
            img *= 1.0 - min(1.0, 2.0 * p) * (self._rows >= skyline[None, :].astype(np.float32))
        np.maximum(img, lines, out=img)
        return img

    def _draw_sand(self, f: Features, amount: float) -> np.ndarray:
        t = self.t
        self._sand_scale += (1.6 + 3.5 * f.pitch - self._sand_scale) * 0.02
        s = self._sand_scale
        u = self._cols4 * s * self._aspect + 0.04 * t
        v = self._rows4 * s - 0.03 * t
        n = 0.7 * value_noise(self._lattice, u, v) + 0.3 * value_noise(self._lattice, u * 2.3 + 5.1, v * 2.3 + 0.07 * t)
        width = 0.018 + 0.035 * f.sustain * amount
        nodal = np.exp(-(((n - 0.5) / width) ** 2))
        density = resize_bilinear(nodal * (0.2 + 0.8 * f.level) * (0.3 + 0.6 * amount), self.H, self.W)
        return (density > self._stipple).astype(np.float32)

    # -- front: ink, blooms, ribbon ----------------------------------------------------

    def _draw_front(self, f: Features) -> np.ndarray | None:
        ax = self._aspect
        X, Y = self._cols4 * ax, self._rows4
        img = None
        bp = self._presence["ink"]
        if bp > 0.02:
            t = self.t
            warp = 0.16 * (0.6 + 0.8 * f.sustain)
            wu = value_noise(self._lattice, X * 2.2 + t * 0.07, Y * 2.2 - t * 0.05) - 0.5
            wv = value_noise(self._lattice, X * 2.2 + 7.3 - t * 0.06, Y * 2.2 + 3.1 + t * 0.04) - 0.5
            Xw, Yw = X + wu * warp * ax, Y + wv * warp
            fld = np.zeros((self.h4, self.w4), dtype=np.float32)
            for k in range(self._blob_n):
                ph, sp, home = self._blob_phase[k], self._blob_speed[k], self._blob_home[k]
                cx = (home[0] + 0.2 * math.sin(t * sp[0] * 2 * math.pi + ph[0]) + 0.06 * math.sin(t * 0.7 + ph[2])) * ax
                cy = home[1] + 0.2 * math.sin(t * sp[1] * 2 * math.pi + ph[1]) + 0.05 * math.cos(t * 0.5 + ph[3])
                r = 0.15 * self._blob_size * (0.45 + 0.7 * f.sustain) * (0.6 + 0.4 * f.level) * (1 + 0.2 * self._pulse + 0.15 * f.kick_env) * math.sqrt(bp)
                fld += (r * r) / ((Xw - cx) ** 2 + (Yw - cy) ** 2 + 1e-4)
            img = resize_bilinear(smoothstep(0.85, 1.15, fld), self.H, self.W)
        if self._blooms:
            blooms = np.zeros((self.h4, self.w4), dtype=np.float32)
            for b in self._blooms:
                rise = 1.0 - math.exp(-b.age / 0.045)
                body = 5.0 * rise * math.exp(-b.age / b.life) * b.strength
                if body < 0.25:
                    continue
                rad = b.size * (0.6 + 0.4 * rise + 0.3 * (1.0 - math.exp(-b.age / 0.6)))
                dx, dy = X - b.x * ax, Y - b.y
                d = np.sqrt(dx * dx + dy * dy)
                th = np.arctan2(dy, dx)
                r_eff = rad * (1.0 + 0.07 * np.sin(3 * th + b.phase[0]) + 0.04 * np.sin(5 * th + b.phase[1]))
                blooms = self.xor(blooms, np.clip((1.0 - d / r_eff) * body, 0.0, 1.0))
            grains = (resize_bilinear(blooms, self.H, self.W) >= self._stipple).astype(np.float32)
            img = grains if img is None else self.xor(img, grains)
        wp = self._presence["ribbon"]
        if wp > 0.02:
            rib = self._draw_ribbon(f, wp)
            img = rib if img is None else self.xor(img, rib)
        return img

    def _draw_ribbon(self, f: Features, wp: float) -> np.ndarray:
        H, W = self.H, self.W
        out = np.zeros((H, W), dtype=np.float32)
        xs = np.linspace(0.0, 1.0, W, dtype=np.float32)
        t, n, ph = self.t, self._strands, self._wave_phase
        self._ribbon_y += (H * (0.78 - 0.56 * f.pitch) - self._ribbon_y) * 0.05
        amp = H * 0.09 * (0.35 + f.mid) * (0.4 + 0.6 * wp) * (1 + 0.3 * self._pulse)
        gap = H * (0.012 + 0.012 * wp)
        twist = 0.25 + 0.9 * f.brightness
        for k in range(n):
            j = k - (n - 1) / 2
            y = (self._ribbon_y + j * gap + amp * (0.65 * np.sin(2 * math.pi * (self._wave_freq * xs + 0.11 * t) + ph[0] + j * twist * 0.35)
                                                   + 0.35 * np.sin(2 * math.pi * (1.7 * self._wave_freq * xs - 0.07 * t) + ph[1] - j * twist * 0.2)))
            yi = np.floor(y).astype(np.int32)
            lo, hi = _line_top(yi), yi + self._thick - 1
            r0, r1 = max(0, int(lo.min())), min(H - 1, int(hi.max()))
            if r1 < r0:
                continue
            rows = np.arange(r0, r1 + 1, dtype=np.int32)[:, None]
            out[r0:r1 + 1][(rows >= lo[None, :]) & (rows <= hi[None, :])] = 1.0
        return out

    # -- events ------------------------------------------------------------------------

    def _anchor(self) -> tuple[float, float]:
        free = [i for i in range(len(self._anchors)) if self.t - self._recent.get(i, -9.0) > 0.6]
        i = int(self.rng.choice(free)) if free else int(self.rng.integers(len(self._anchors)))
        self._recent[i] = self.t
        return float(self._anchors[i][0]), float(self._anchors[i][1])

    def _pick(self, lean: dict) -> str | None:
        names = [k for k, v in lean.items() if v > 0]
        if not names:
            return None
        p = np.array([lean[k] for k in names]) + 1e-9
        return names[int(self.rng.choice(len(names), p=p / p.sum()))]

    def _on_kick(self, f: Features, calm: float, stark: float, coverage: float) -> None:
        k = min(1.0, f.kick)
        w = self._mix_w["kick"]
        blooms, ripples = self.on("blooms"), self.on("ripples")
        choice = self._pick({"bloom": w["bloom"] * (0.6 + 0.8 * calm) * blooms, "ripple": w["ripple"] * (0.4 + 1.2 * stark) * ripples,
                             "pulse": w["pulse"] * (0.7 + 0.5 * calm) * ripples})
        if choice is None:
            return
        do = {choice}
        if choice == "pulse" and blooms:
            do.add("bloom")
        if k > 0.55 and w["ripple"] > 0.12 and ripples:
            do.add("ripple")
        if k > 0.8 and ripples:
            do.add("pulse")
        x, y = self._anchor()
        if "bloom" in do and len(self._blooms) < MAX_BLOOMS:
            b = _Event(x, y, strength=min(1.0, 0.55 + 0.6 * k), life=0.16 + 0.2 * calm, size=(0.09 + 0.2 * k) * (0.55 + 0.7 * coverage))
            b.phase = self.rng.uniform(0, 2 * math.pi, 4).astype(np.float32)
            self._blooms.append(b)
        if "ripple" in do and len(self._ripples) < MAX_RIPPLES:
            self._ripples.append(_Event(x, y, strength=k * (0.5 + 0.5 * coverage), life=0.9))
        if "pulse" in do:
            self._pulse = max(self._pulse, k * (0.5 + 0.5 * coverage))

    def _on_hit(self, f: Features, calm: float, stark: float, coverage: float) -> None:
        h = min(1.0, f.hit)
        w = self._mix_w["bright"]
        sharp = f.brightness
        choice = self._pick({
            "grain": w["grain"] * (0.6 + 0.8 * calm) * (0.6 + 0.8 * f.noisiness) * self.on("grain"),
            "lines": w["lines"] * (0.4 + 1.2 * stark) * (0.5 + sharp) * self.on("hairlines"),
            "shards": w["shards"] * (0.3 + 1.0 * stark) * (0.2 + 1.3 * sharp) * self.on("shards"),
            "texture": w["texture"] * (0.8 + 0.4 * calm) * self.on("grain"),
        })
        if choice == "grain":
            self._grain = max(self._grain, h * (0.6 + 0.6 * coverage))
        elif choice == "texture":
            self._texture = max(self._texture, h * (0.6 + 0.6 * coverage))
        elif choice == "lines":
            for _ in range(1 + int(h * 2.5 * (0.5 + stark))):
                if len(self._lines) >= MAX_LINES:
                    break
                self._lines.append(_Event(float(self.rng.uniform(0.05, 0.95)), float(self.rng.uniform(0.05, 0.95)),
                                          strength=min(1.0, 0.5 + h), life=float(self.rng.uniform(0.1, 0.28)),
                                          size=float(self.rng.uniform(0.15, 0.7)) * (0.6 + 0.6 * coverage),
                                          angle=self._line_angle + float(self.rng.normal(0, 0.03))))
        elif choice == "shards":
            for _ in range(1 + int(h * 1.5)):
                if len(self._shards) >= MAX_SHARDS:
                    break
                x, y = self._anchor()
                e = _Event(x + float(self.rng.normal(0, 0.04)), y + float(self.rng.normal(0, 0.04)), strength=min(1.0, 0.5 + h),
                           life=float(self.rng.uniform(0.18, 0.35)), size=(0.018 + 0.035 * sharp) * (0.6 + 0.8 * coverage),
                           angle=float(self.rng.uniform(0, 2 * math.pi)), spin=float(self.rng.normal(0, 1.2)))
                e.phase = self.rng.uniform(-0.3, 0.3, 4).astype(np.float32)
                self._shards.append(e)

    def _draw_sharp(self, img: np.ndarray, dt: float) -> np.ndarray:
        if not self._lines and not self._shards:
            return img
        H, W = self.H, self.W
        top = np.zeros((H, W), dtype=np.float32)
        for e in self._lines:
            length = e.size * max(W, H) * max(0.0, 1.0 - e.age / e.life) * min(1.0, 0.4 + e.age / 0.03) * e.strength
            if length < 2:
                continue
            s = np.linspace(-length / 2, length / 2, int(length) + 2, dtype=np.float32)
            xs = np.round(e.x * W + math.cos(e.angle) * s).astype(np.int32)
            ys = np.round(e.y * H + math.sin(e.angle) * s).astype(np.int32)
            ok = (xs >= 0) & (xs < W) & (ys >= 0) & (ys < H)
            top[ys[ok], xs[ok]] = 1.0
        for e in self._shards:
            e.angle += e.spin * dt
            r = e.size * H * max(0.0, 1.0 - e.age / e.life) * min(1.0, 0.3 + e.age / 0.03)
            if r < 1.5:
                continue
            cx, cy = e.x * W, e.y * H
            angs = e.angle + np.array([0.0, 2.35 + e.phase[0], 3.93 + e.phase[1]])
            rads = r * np.array([1.0, 0.32 + 0.15 * e.phase[2], 0.32 + 0.15 * e.phase[3]])
            vx, vy = cx + np.cos(angs) * rads, cy + np.sin(angs) * rads
            x0, x1 = max(0, int(vx.min())), min(W - 1, int(vx.max()) + 1)
            y0, y1 = max(0, int(vy.min())), min(H - 1, int(vy.max()) + 1)
            if x1 <= x0 or y1 <= y0:
                continue
            px = np.arange(x0, x1 + 1, dtype=np.float32)[None, :] + 0.5
            py = np.arange(y0, y1 + 1, dtype=np.float32)[:, None] + 0.5
            s = [(vx[(j + 1) % 3] - vx[j]) * (py - vy[j]) - (vy[(j + 1) % 3] - vy[j]) * (px - vx[j]) for j in range(3)]
            inside = ((s[0] >= 0) & (s[1] >= 0) & (s[2] >= 0)) | ((s[0] <= 0) & (s[1] <= 0) & (s[2] <= 0))
            top[y0:y1 + 1, x0:x1 + 1][inside] = 1.0
        return self.xor(img, top)

    def _draw_grain(self, img: np.ndarray) -> np.ndarray:
        g, tx = self._grain, self._texture
        if g < 0.02 and tx < 0.02:
            return img
        H, W = self.H, self.W
        gs = self._grain_size
        rnd = self.rng.random(((H + gs - 1) // gs, (W + gs - 1) // gs), dtype=np.float32)
        if gs > 1:
            rnd = np.repeat(np.repeat(rnd, gs, axis=0), gs, axis=1)[:H, :W]
        specks = np.zeros((H, W), dtype=np.float32)
        if g >= 0.02:
            specks = (rnd < g * 0.05).astype(np.float32)
        if tx >= 0.02:
            edge = np.abs(img - np.roll(img, 1, axis=1)) + np.abs(img - np.roll(img, 1, axis=0))
            specks = np.maximum(specks, ((edge > 0.35) & (rnd > 1.0 - 0.55 * tx)).astype(np.float32))
        return self.xor(img, specks)

    def _displace(self, img: np.ndarray) -> np.ndarray:
        active = [r for r in self._ripples if r.age < r.life]
        z = 0.035 * self._pulse
        if not active and z < 0.002:
            return img
        H, W = self.H, self.W
        h2, w2 = (H + 1) // 2, (W + 1) // 2
        X2 = np.arange(w2, dtype=np.float32)[None, :] * 2.0
        Y2 = np.arange(h2, dtype=np.float32)[:, None] * 2.0
        dx = np.zeros((h2, w2), dtype=np.float32)
        dy = np.zeros((h2, w2), dtype=np.float32)
        diag = math.hypot(W, H)
        for r in active:
            ox, oy = X2 - r.x * W, Y2 - r.y * H
            d = np.sqrt(ox * ox + oy * oy) + 1e-3
            radius = diag * 0.75 * (1.0 - math.exp(-r.age / 0.35))
            width = diag * (0.03 + 0.03 * r.age)
            u = (d - radius) / width
            near = np.abs(u) < 3.0
            push = np.zeros_like(d)
            push[near] = r.strength * max(0.0, 1.0 - r.age / r.life) * 0.022 * diag * np.exp(-u[near] ** 2) * np.sin(math.pi * u[near])
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
        for lst in (self._blooms, self._ripples, self._shards, self._lines):
            for e in lst:
                e.age += dt
        self._blooms = [b for b in self._blooms if 5.0 * math.exp(-b.age / b.life) * b.strength >= 0.25]
        self._ripples = [r for r in self._ripples if r.age < r.life]
        self._shards = [s for s in self._shards if s.age < s.life]
        self._lines = [e for e in self._lines if e.age < e.life]
