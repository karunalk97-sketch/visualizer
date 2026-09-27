"""Character mode: the picture is drawn with glyphs instead of dithered pixels.

The screen is a grid of cells. Every cell holds one glyph -- a built-in shape
(circle, square, triangle, ...) or any character you type, in any installed
font (Wingdings and friends work: type the letters, the font draws the
symbols). Brightness decides how bright *and how big* a glyph is drawn, so the
bit-depth setting still means something (2 / 4 / 8 / 16 sizes and grays).

Two ways to choose the glyph for a cell:

* "random"     -- each cell gets a random glyph from your set (stable, and
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

BUILTIN_SHAPES = ["circle", "square", "triangle", "diamond", "plus", "cross", "star", "ring", "dot", "bar", "slash"]
_SS = 4  # supersampling for the built-in shapes


def _shape_surface(name: str, cw: int, ch: int) -> pygame.Surface:
    s = pygame.Surface((cw * _SS, ch * _SS))
    s.fill((0, 0, 0))
    cx, cy = cw * _SS / 2, ch * _SS / 2
    r = 0.42 * min(cw, ch) * _SS
    w = (255, 255, 255)
    if name == "circle":
        pygame.draw.circle(s, w, (cx, cy), r)
    elif name == "dot":
        pygame.draw.circle(s, w, (cx, cy), r * 0.45)
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
        pts = []
        for k in range(10):
            a = -math.pi / 2 + k * math.pi / 5
            rr = r * 1.15 if k % 2 == 0 else r * 0.5
            pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
        pygame.draw.polygon(s, w, pts)
    elif name == "bar":
        pygame.draw.rect(s, w, (cx - r, cy - r * 0.22, 2 * r, r * 0.44))
    elif name == "slash":
        pygame.draw.line(s, w, (cx - r * 0.8, cy + r * 0.9), (cx + r * 0.8, cy - r * 0.9), max(2, int(r * 0.4)))
    else:
        pygame.draw.circle(s, w, (cx, cy), r)
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
    return _coverage(_shape_surface(name, cw, ch), cw, ch)


def char_coverage(char: str, font_name: str, cw: int, ch: int) -> np.ndarray | None:
    """Coverage of one character, fitted and centred in the cell; None if it draws nothing."""
    if not char or char.isspace():
        return None
    size = max(6, int(min(cw, ch) * 1.3))  # fill the cell: thin strokes vanish at small sizes
    font = pygame.font.SysFont(font_name or "segoeui,arial,helvetica,dejavusans", size)
    font.set_bold(min(cw, ch) <= 20)  # bolder strokes stay legible in small cells
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
    return cov if cov.max() > 0.05 else None


def glyph_set(shapes: list[str], chars: str, font_name: str, cell: int) -> list[np.ndarray]:
    """Coverage arrays for every chosen shape and every distinct typed character."""
    glyphs = [shape_coverage(n, cell, cell) for n in shapes if n in BUILTIN_SHAPES]
    seen: set[str] = set()
    for c in chars:
        if c in seen:
            continue
        seen.add(c)
        cov = char_coverage(c, font_name, cell, cell)
        if cov is not None:
            glyphs.append(cov)
    if not glyphs:
        glyphs = [shape_coverage("circle", cell, cell)]
    return glyphs


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

    def configure(self, shapes: list[str], chars: str, font_name: str, cell: int, bits: int) -> None:
        """(Re)build the atlas when any glyph-related setting changed."""
        sig = (tuple(shapes), chars, font_name, cell, bits)
        if sig == self._sig:
            return
        self._sig = sig
        glyphs = glyph_set(shapes, chars, font_name, cell)
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
