from __future__ import annotations

import argparse
import sys

import numpy as np
import pygame

from .analyzer import SpectrumAnalyzer
from .audio_capture import SyntheticSource, WasapiLoopbackSource
from .config import Config
from .palette import PALETTES, get_palette
from .renderer import BitmapRenderer

CHUNK_SIZE = 1024
PALETTE_NAMES = list(PALETTES.keys()) + ["custom"]
BIT_DEPTHS = [1, 2, 4, 8]


def build_source(demo: bool):
    if demo or sys.platform != "win32":
        if not demo:
            print("System-audio loopback needs pyaudiowpatch and only works on Windows; "
                  "falling back to --demo mode (a synthetic sweep tone).")
        return SyntheticSource()
    return WasapiLoopbackSource()


def main() -> None:
    parser = argparse.ArgumentParser(description="Retro bitmap audio visualizer")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--demo", action="store_true", help="use a synthetic test tone instead of system audio")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    source = build_source(args.demo)

    pygame.init()
    renderer = BitmapRenderer(cfg.grid_width, cfg.grid_height, cfg.window_scale)
    analyzer = SpectrumAnalyzer(
        sample_rate=source.sample_rate,
        num_bands=cfg.num_bands,
        decay=cfg.decay,
        gain=cfg.gain,
    )
    clock = pygame.time.Clock()

    palette_idx = PALETTE_NAMES.index(cfg.palette) if cfg.palette in PALETTE_NAMES else 0
    depth_idx = BIT_DEPTHS.index(cfg.bit_depth) if cfg.bit_depth in BIT_DEPTHS else 1

    frame_iter = source.frames(CHUNK_SIZE)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_p:
                    palette_idx = (palette_idx + 1) % len(PALETTE_NAMES)
                    cfg.palette = PALETTE_NAMES[palette_idx]
                elif event.key == pygame.K_b:
                    depth_idx = (depth_idx + 1) % len(BIT_DEPTHS)
                    cfg.bit_depth = BIT_DEPTHS[depth_idx]
                elif event.key == pygame.K_d:
                    cfg.dither = not cfg.dither
                elif event.key == pygame.K_m:
                    cfg.mode = "waveform" if cfg.mode == "bars" else "bars"
                elif event.key == pygame.K_s:
                    cfg.save(args.config)
                    print(f"saved settings to {args.config}")

        samples = next(frame_iter)
        colors = get_palette(cfg.palette, cfg.bit_depth, cfg.custom_colors)

        if cfg.mode == "bars":
            bands = analyzer.process(samples)
            renderer.render_bars(bands, colors, cfg.dither)
        else:
            mono = samples.mean(axis=1) if samples.ndim > 1 else samples
            renderer.render_waveform(mono, colors, cfg.dither)

        clock.tick(cfg.fps)

    pygame.quit()


if __name__ == "__main__":
    main()
