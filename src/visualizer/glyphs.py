"""Character mode: the picture is drawn with glyphs instead of dithered pixels.

The screen is a grid of cells and every cell holds one glyph from the sets you
switch on:

* Shapes   -- circle, square, triangle, diamond, plus, cross, star, ring;
* Symbols  -- drawn dingbats (heart, arrow, check, flower, moon, sun, bolt, note,
              drop, leaf), a Wingdings-style set that looks the same on every OS;
* ASCII    -- . : - = + * # % @
* Binary   -- 0 and 1

Brightness decides how bright *and how big* a glyph is drawn, so the bit-depth
setting still means something (2 / 4 / 8 / 16 sizes and grays).

Two ways to choose the glyph for a cell:

* "random"     -- each cell gets a random glyph from the chosen sets (stable, and
                  reshuffled whenever the visualizer reshuffles, e.g. a new song);
* "brightness" -- glyphs are sorted by how much ink they use and brighter areas
                  get denser glyphs, like classic ASCII art.

Every glyph is drawn once per brightness level into an atlas, so a frame is one
vectorised lookup rather than thousands of blits.
"""
from __future__ import annotations

import math

import numpy as np
import pygame

from .palette import quantize_indices

SHAPE_NAMES = ["circle", "square", "triangle", "diamond", "plus", "cross", "star", "ring"]
SYMBOL_NAMES = ["heart", "arrow", "check", "flower", "moon", "sun", "bolt", "note", "drop", "leaf"]
ASCII_CHARS = ".:-=+*#%@"
BINARY_CHARS = "01"
MONO_FONTS = "consolas,couriernew,menlo,dejavusansmono,monospace"
_SS = 4  # supersampling for the drawn glyphs
_TARGET = 8  # small sets are repeated up to about this many entries, so each set gets a fair share


def _canvas(cw: int, ch: int) -> tuple[pygame.Surface, float, float, float]:
    s = pygame.Surface((cw * _SS, ch * _SS))
    s.fill((0, 0, 0))
    return s, cw * _SS / 2, ch * _SS / 2, 0.42 * min(cw, ch) * _SS


def _draw_shape(name: str, cw: int, ch: int) -> pygame.Surface:
    s, cx, cy, r = _canvas(cw, ch)
    w = (255, 255, 255)
    if name == "circle":
        pygame.draw.circle(s, w, (cx, cy), r)
    elif name == "ring":
        pygame.draw.circle(s, w, (cx, cy), r, max(2, int(r * 0.32)))
    elif name == "square":
        pygame.draw.rect(s, w, (cx - r * 0.85, cy - r * 0.85, r * 1.7, r * 1.7))
    elif name == "diamond":
        pygame.draw.polygon(s, w, [(cx, cy - r * 1.15), (cx + r * 1.15, cy), (cx, cy + r * 1.15), (cx - r * 1.15, cy)])
    elif name == "triangle":
        pygame.draw.polygon(s, w, [(cx, cy - r * 1.05), (cx + r * 1.05, cy + r * 0.85), (cx - r * 1.05, cy + r * 0.85)])
    elif name == "plus":
        t = r * 0.3
        pygame.draw.rect(s, w, (cx - r, cy - t, 2 * r, 2 * t))
        pygame.draw.rect(s, w, (cx - t, cy - r, 2 * t, 2 * r))
    elif name == "cross":
        pygame.draw.line(s, w, (cx - r * 0.85, cy - r * 0.85), (cx + r * 0.85, cy + r * 0.85), max(2, int(r * 0.42)))
        pygame.draw.line(s, w, (cx - r * 0.85, cy + r * 0.85), (cx + r * 0.85, cy - r * 0.85), max(2, int(r * 0.42)))
    elif name == "star":
        pygame.draw.polygon(s, w, _star_points(cx, cy, r * 1.15, r * 0.5, 5))
    return s


def _star_points(cx: float, cy: float, r_out: float, r_in: float, n: int) -> list[tuple[float, float]]:
    pts = []
    for k in range(2 * n):
        a = -math.pi / 2 + k * math.pi / n
        rr = r_out if k % 2 == 0 else r_in
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    return pts


def _draw_symbol(name: str, cw: int, ch: int) -> pygame.Surface:
    s, cx, cy, r = _canvas(cw, ch)
    w, k = (255, 255, 255), (0, 0, 0)
    if name == "heart":
        pygame.draw.circle(s, w, (cx - r * 0.5, cy - r * 0.3), r * 0.58)
        pygame.draw.circle(s, w, (cx + r * 0.5, cy - r * 0.3), r * 0.58)
        pygame.draw.polygon(s, w, [(cx - r * 1.06, cy - r * 0.05), (cx + r * 1.06, cy - r * 0.05), (cx, cy + r * 1.1)])
    elif name == "arrow":
        pygame.draw.polygon(s, w, [(cx - r, cy - r * 0.27), (cx + r * 0.15, cy - r * 0.27), (cx + r * 0.15, cy - r * 0.8),
                                   (cx + r * 1.05, cy), (cx + r * 0.15, cy + r * 0.8), (cx + r * 0.15, cy + r * 0.27),
                                   (cx - r, cy + r * 0.27)])
    elif name == "check":
        pygame.draw.lines(s, w, False, [(cx - r * 0.85, cy + r * 0.05), (cx - r * 0.25, cy + r * 0.65), (cx + r * 0.9, cy - r * 0.7)],
                          max(2, int(r * 0.4)))
    elif name == "flower":
        for k_ in range(6):
            a = k_ * math.pi / 3
            pygame.draw.circle(s, w, (cx + r * 0.58 * math.cos(a), cy + r * 0.58 * math.sin(a)), r * 0.42)
        pygame.draw.circle(s, k, (cx, cy), r * 0.2)
    elif name == "moon":
        pygame.draw.circle(s, w, (cx, cy), r)
        pygame.draw.circle(s, k, (cx + r * 0.5, cy - r * 0.12), r * 0.85)
    elif name == "sun":
        pygame.draw.circle(s, w, (cx, cy), r * 0.5)
        for k_ in range(8):
            a = k_ * math.pi / 4
            pygame.draw.line(s, w, (cx + r * 0.75 * math.cos(a), cy + r * 0.75 * math.sin(a)),
                             (cx + r * 1.1 * math.cos(a), cy + r * 1.1 * math.sin(a)), max(2, int(r * 0.2)))
    elif name == "bolt":
        pygame.draw.polygon(s, w, [(cx + r * 0.2, cy - r * 1.1), (cx - r * 0.6, cy + r * 0.1), (cx - r * 0.05, cy + r * 0.1),
                                   (cx - r * 0.25, cy + r * 1.1), (cx + r * 0.65, cy - r * 0.2), (cx + r * 0.1, cy - r * 0.2)])
    elif name == "note":
        pygame.draw.circle(s, w, (cx - r * 0.4, cy + r * 0.62), r * 0.42)
        pygame.draw.rect(s, w, (cx - r * 0.06, cy - r * 0.95, r * 0.22, r * 1.6))
        pygame.draw.polygon(s, w, [(cx + r * 0.16, cy - r * 0.95), (cx + r * 0.85, cy - r * 0.4), (cx + r * 0.16, cy - r * 0.35)])
    elif name == "drop":
        pygame.draw.circle(s, w, (cx, cy + r * 0.33), r * 0.62)
        pygame.draw.polygon(s, w, [(cx, cy - r * 1.05), (cx - r * 0.58, cy + r * 0.2), (cx + r * 0.58, cy + r * 0.2)])
    elif name == "leaf":
        pts = []
        for k_ in range(24):
            t = k_ / 23
            x = (t - 0.5) * 2 * r * 1.05
            bulge = math.sin(math.pi * t) * r * 0.55
            pts.append((x, -bulge))
        pts += [(x, -y) for x, y in reversed(pts)]
        rot = math.radians(-40)
        pygame.draw.polygon(s, w, [(cx + x * math.cos(rot) - y * math.sin(rot), cy + x * math.sin(rot) + y * math.cos(rot)) for x, y in pts])
    return s


def _coverage(surface: pygame.Surface, cw: int, ch: int) -> np.ndarray:
    """A white-on-black surface -> (ch, cw) float coverage 0..1, block-averaged if supersampled."""
    a = pygame.surfarray.array3d(surface)[:, :, 0].astype(np.float32).T / 255.0  # (h, w)
    h, w = a.shape
    if (w, h) != (cw, ch):
        fy, fx = h // ch, w // cw
        a = a[: ch * fy, : cw * fx].reshape(ch, fy, cw, fx).mean(axis=(1, 3))
    return a


def shape_coverage(name: str, cw: int, ch: int) -> np.ndarray:
    return _coverage(_draw_shape(name, cw, ch), cw, ch)


def symbol_coverage(name: str, cw: int, ch: int) -> np.ndarray:
    return _coverage(_draw_symbol(name, cw, ch), cw, ch)


def char_coverage(char: str, font_name: str, cw: int, ch: int) -> np.ndarray | None:
    """Coverage of one character, fitted and centred in the cell; None if it draws nothing."""
    if not char or char.isspace():
        return None
    size = max(6, int(min(cw, ch) * 1.3))  # fill the cell: thin strokes vanish at small sizes
    font = pygame.font.SysFont(font_name or MONO_FONTS, size)
    font.set_bold(min(cw, ch) <= 20)       # bolder strokes stay legible in small cells
    try:
        img = font.render(char, True, (255, 255, 255))  # 32-bit with alpha, so it can be scaled
    except Exception:
        return None
    iw, ih = img.get_size()
    if iw > cw or ih > ch:  # too big for the cell: shrink to fit
        k = min(cw / iw, ch / ih)
        img = pygame.transform.smoothscale(img, (max(1, int(iw * k)), max(1, int(ih * k))))
        iw, ih = img.get_size()
    canvas = pygame.Surface((cw, ch))
    canvas.fill((0, 0, 0))
    canvas.blit(img, ((cw - iw) // 2, (ch - ih) // 2))
    cov = _coverage(canvas, cw, ch)
    if cov.max() <= 0.05:
        return None
    return np.clip(cov * 1.6, 0.0, 1.0)  # antialiased edges are gray; firm them up so text reads as white


def _set_glyphs(name: str, cell: int) -> list[np.ndarray]:
    if name == "shapes":
        return [shape_coverage(n, cell, cell) for n in SHAPE_NAMES]
    if name == "symbols":
        return [symbol_coverage(n, cell, cell) for n in SYMBOL_NAMES]
    chars = ASCII_CHARS if name == "ascii" else BINARY_CHARS if name == "binary" else ""
    out = [char_coverage(c, MONO_FONTS, cell, cell) for c in chars]
    return [g for g in out if g is not None]


def glyph_set(sets: list[str], cell: int) -> list[np.ndarray]:
    """Coverage arrays for every glyph in the chosen sets. Small sets (Binary has
    just two) are repeated so each chosen set gets a fair share of the picture."""
    glyphs: list[np.ndarray] = []
    for name in sets:
        g = _set_glyphs(name, cell)
        if g:
            glyphs += g * max(1, round(_TARGET / len(g)))
    return glyphs or [shape_coverage("circle", cell, cell)]


def build_atlas(glyphs: list[np.ndarray], levels: int, cell: int) -> np.ndarray:
    """(G, levels, cell, cell) uint8. Level 0 is blank; higher levels are brighter and bigger."""
    atlas = np.zeros((len(glyphs), levels, cell, cell), dtype=np.uint8)
    for g, cov in enumerate(glyphs):
        surf = pygame.surfarray.make_surface((np.repeat(cov.T[:, :, None], 3, axis=2) * 255).astype(np.uint8))
        for lvl in range(1, levels):
            b = lvl / (levels - 1)
            scale = 0.5 + 0.5 * b
            size = max(1, int(round(cell * scale)))
            small = pygame.transform.smoothscale(surf, (size, size)) if size != cell else surf
            arr = pygame.surfarray.array3d(small)[:, :, 0].astype(np.float32).T / 255.0
            canvas = np.zeros((cell, cell), dtype=np.float32)
            off = (cell - size) // 2
            canvas[off:off + size, off:off + size] = arr
            atlas[g, lvl] = np.clip(canvas * 255 * (0.4 + 0.6 * b), 0, 255).astype(np.uint8)
    return atlas


class GlyphField:
    """Turns a per-cell brightness map into a full-resolution grayscale image of glyphs."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)
        self._sig: tuple | None = None
        self._atlas: np.ndarray | None = None
        self._order: np.ndarray | None = None   # glyph indices sorted by ink, least first
        self._assign: np.ndarray | None = None  # (rows, cols) random glyph per cell
        self._assign_key: tuple | None = None

    @property
    def glyph_count(self) -> int:
        return 0 if self._atlas is None else self._atlas.shape[0]

    def configure(self, sets: list[str], cell: int, bits: int) -> None:
        """(Re)build the atlas when any glyph-related setting changed."""
        sig = (tuple(sets), cell, bits)
        if sig == self._sig:
            return
        self._sig = sig
        glyphs = glyph_set(list(sets), cell)
        self._atlas = build_atlas(glyphs, 2 ** bits, cell)
        ink = self._atlas[:, -1].reshape(len(glyphs), -1).sum(axis=1)
        self._order = np.argsort(ink, kind="stable")
        self._assign_key = None

    def reshuffle(self) -> None:
        self._assign_key = None

    def render(self, intensity: np.ndarray, mapping: str = "random") -> np.ndarray:
        """intensity: (rows, cols) floats 0..1 -> (rows*cell, cols*cell) uint8 image."""
        assert self._atlas is not None, "call configure() first"
        g, levels, cell, _ = self._atlas.shape
        rows, cols = intensity.shape
        lvl = quantize_indices(intensity, levels)  # dithered, so glyph density shows gradients
        if mapping == "brightness" and g > 1:
            pos = np.minimum((np.clip(intensity, 0, 1) * g).astype(np.intp), g - 1)
            idx = self._order[pos]
        else:
            key = (rows, cols, g)
            if self._assign_key != key or self._assign is None:
                self._assign = self._rng.integers(0, g, size=(rows, cols)).astype(np.intp)
                self._assign_key = key
            idx = self._assign
        tiles = self._atlas[idx, lvl]                      # (rows, cols, cell, cell)
        return tiles.transpose(0, 2, 1, 3).reshape(rows * cell, cols * cell)
