"""Draws into a low-resolution framebuffer (the "bitmap"), scales it up with
nearest-neighbor to fill the window above a slim status bar, producing the
chunky pixel look. All color decisions go through palette.quantize. The status
bar is drawn at real screen resolution in a normal system font so it stays
legible however chunky the picture is.
"""
from __future__ import annotations

import numpy as np
import pygame

from . import palette as palette_mod

BLACK = (0, 0, 0)
TEXT = (205, 205, 205)
TEXT_DIM = (125, 125, 125)
FONT_CANDIDATES = "segoeui,helveticaneue,helvetica,arial,dejavusans,sans"


def bar_height(screen_height: int) -> int:
    """Slim status bar: ~3.5% of the window height, never tiny or huge."""
    return int(np.clip(round(screen_height * 0.035), 24, 40))


class BitmapRenderer:
    def __init__(self, grid_width: int, grid_height: int, screen_width: int, screen_height: int) -> None:
        pygame.font.init()
        self._fonts: dict[int, pygame.font.Font] = {}
        self._surf: pygame.Surface | None = None
        self.resize(grid_width, grid_height, screen_width, screen_height)

    def resize(self, grid_width: int, grid_height: int, screen_width: int, screen_height: int) -> None:
        """Call after the window is created or resized. The grid covers only
        the picture area (window minus the status bar)."""
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.get_surface()
        self.bar_h = bar_height(screen_height)
        self.field_h = max(1, screen_height - self.bar_h)
        self._font = self._font_for(round(self.bar_h * 0.52))

    def _font_for(self, px: int) -> pygame.font.Font:
        if px not in self._fonts:
            self._fonts[px] = pygame.font.SysFont(FONT_CANDIDATES, px)
        return self._fonts[px]

    def _draw_bar(self, left_text: str, right_text: str) -> None:
        bar = pygame.Rect(0, self.field_h, self.screen_width, self.bar_h)
        self.screen.fill(BLACK, bar)
        pad = max(10, self.bar_h // 2)
        if left_text:
            img = self._font.render(left_text, True, TEXT)
            self.screen.blit(img, (pad, bar.centery - img.get_height() // 2))
        if right_text:
            img = self._font.render(right_text, True, TEXT_DIM)
            self.screen.blit(img, (self.screen_width - img.get_width() - pad, bar.centery - img.get_height() // 2))

    def _present(self, surf: pygame.Surface, left_text: str, right_text: str) -> None:
        scaled = pygame.transform.scale(surf, (self.screen_width, self.field_h))
        self.screen.blit(scaled, (0, 0))
        self._draw_bar(left_text, right_text)
        pygame.display.flip()

    def _indexed_surface(self, intensity: np.ndarray, colors) -> pygame.Surface:
        """8-bit palettised surface: a third of the memory of RGB, much faster to scale."""
        idx = palette_mod.quantize_indices(intensity, len(colors))
        h, w = idx.shape
        if self._surf is None or self._surf.get_size() != (w, h):
            self._surf = pygame.Surface((w, h), depth=8)
        self._surf.set_palette(colors)
        pygame.surfarray.blit_array(self._surf, idx.T)
        return self._surf

    def render_field(self, intensity: np.ndarray, colors, left_text: str = "", right_text: str = "") -> None:
        self._present(self._indexed_surface(intensity, colors), left_text, right_text)

    def render_waveform(self, samples: np.ndarray, colors, left_text: str = "", right_text: str = "") -> None:
        h, w = self.grid_height, self.grid_width
        intensity = np.zeros((h, w), dtype=np.float32)
        if len(samples) > 0:
            xs = np.linspace(0, len(samples) - 1, w).astype(np.int32)
            ys = samples[xs]
            rows = np.clip(((1 - (ys * 0.5 + 0.5)) * (h - 1)).astype(np.int32), 0, h - 1)
            intensity[rows, np.arange(w)] = 1.0
        self._present(self._indexed_surface(intensity, colors), left_text, right_text)
