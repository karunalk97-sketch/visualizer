"""Draws into a small low-resolution framebuffer (the "bitmap") and scales it
up with nearest-neighbor to fill the real screen, producing the chunky,
retro-pixel look. All color decisions go through palette.quantize.
"""
from __future__ import annotations

import numpy as np
import pygame

from . import palette as palette_mod

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


class BitmapRenderer:
    def __init__(self, grid_width: int, grid_height: int, screen_width: int, screen_height: int) -> None:
        pygame.font.init()
        self.resize(grid_width, grid_height, screen_width, screen_height)

    def resize(self, grid_width: int, grid_height: int, screen_width: int, screen_height: int) -> None:
        """Call after the window is created or resized."""
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.get_surface()

        font_px = max(10, grid_height // 12)
        self._bar_h = font_px + 4
        self._font = pygame.font.Font(None, font_px)

    def _present(self, surf: pygame.Surface) -> None:
        scaled = pygame.transform.scale(surf, (self.screen_width, self.screen_height))
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def _blit_pixel_text(self, surf: pygame.Surface, text: str, right_align: bool) -> None:
        if not text:
            return
        rendered = self._font.render(text, False, WHITE)  # antialias=False -> blocky/pixelated glyphs
        y = self.grid_height - self._bar_h + 1
        if right_align:
            x = max(2, self.grid_width - rendered.get_width() - 2)
        else:
            x = 2
        surf.blit(rendered, (x, y))

    def render_field(self, intensity: np.ndarray, colors, left_text: str = "", right_text: str = "") -> None:
        rgb = palette_mod.quantize(intensity, colors)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))

        if left_text or right_text:
            pygame.draw.rect(surf, BLACK, (0, self.grid_height - self._bar_h, self.grid_width, self._bar_h))
            self._blit_pixel_text(surf, left_text, right_align=False)
            self._blit_pixel_text(surf, right_text, right_align=True)

        self._present(surf)

    def render_waveform(self, samples: np.ndarray, colors) -> None:
        h, w = self.grid_height, self.grid_width
        intensity = np.zeros((h, w), dtype=np.float32)
        if len(samples) > 0:
            xs = np.linspace(0, len(samples) - 1, w).astype(np.int32)
            ys = samples[xs]
            rows = np.clip(((1 - (ys * 0.5 + 0.5)) * (h - 1)).astype(np.int32), 0, h - 1)
            intensity[rows, np.arange(w)] = 1.0
        rgb = palette_mod.quantize(intensity, colors)
        self._present(pygame.surfarray.make_surface(rgb.swapaxes(0, 1)))
