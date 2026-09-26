"""Draws into a small low-resolution framebuffer (the "bitmap") and scales it
up with nearest-neighbor to produce the chunky, retro-pixel look. All actual
color decisions go through palette.quantize, so bit depth/dithering apply
uniformly regardless of what's being drawn.
"""
from __future__ import annotations

import numpy as np
import pygame

from . import palette as palette_mod


class BitmapRenderer:
    def __init__(self, grid_width: int, grid_height: int, window_scale: int) -> None:
        self.grid_width = grid_width
        self.grid_height = grid_height
        self.window_scale = window_scale
        self.screen = pygame.display.set_mode(
            (grid_width * window_scale, grid_height * window_scale)
        )
        pygame.display.set_caption("Audio Visualizer")

    def _blit_intensity(self, intensity: np.ndarray, colors, dither: bool) -> None:
        rgb = palette_mod.quantize(intensity, colors, dither)  # (H, W, 3) uint8
        # pygame surfarray wants (W, H, 3)
        surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        scaled = pygame.transform.scale(
            surf, (self.grid_width * self.window_scale, self.grid_height * self.window_scale)
        )
        self.screen.blit(scaled, (0, 0))
        pygame.display.flip()

    def render_bars(self, band_values: np.ndarray, colors, dither: bool) -> None:
        h, w = self.grid_height, self.grid_width
        intensity = np.zeros((h, w), dtype=np.float32)
        n_bands = len(band_values)
        col_width = max(1, w // n_bands)
        for i, level in enumerate(band_values):
            bar_h = int(level * h)
            if bar_h <= 0:
                continue
            x0 = i * col_width
            x1 = min(w, x0 + col_width)
            intensity[h - bar_h:h, x0:x1] = 1.0
        self._blit_intensity(intensity, colors, dither)

    def render_waveform(self, samples: np.ndarray, colors, dither: bool) -> None:
        h, w = self.grid_height, self.grid_width
        intensity = np.zeros((h, w), dtype=np.float32)
        if len(samples) == 0:
            self._blit_intensity(intensity, colors, dither)
            return
        xs = np.linspace(0, len(samples) - 1, w).astype(np.int32)
        ys = samples[xs]
        rows = ((1 - (ys * 0.5 + 0.5)) * (h - 1)).astype(np.int32)
        rows = np.clip(rows, 0, h - 1)
        intensity[rows, np.arange(w)] = 1.0
        self._blit_intensity(intensity, colors, dither)
