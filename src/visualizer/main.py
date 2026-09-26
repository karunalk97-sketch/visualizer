from __future__ import annotations

import argparse
import sys

import pygame

from .analyzer import SpectrumAnalyzer
from .audio_capture import SyntheticSource, WasapiLoopbackSource
from .config import Config
from .now_playing import NowPlayingWatcher
from .palette import grayscale_palette
from .renderer import BitmapRenderer
from .spectral_field import SpectralField, resize_nearest

CHUNK_SIZE = 1024
BIT_DEPTHS = [1, 2, 3, 4]
MODES = ["field", "waveform"]


def build_source(demo: bool):
    if demo or sys.platform != "win32":
        if not demo:
            print("System-audio loopback needs pyaudiowpatch and only works on Windows; "
                  "falling back to --demo mode (a synthetic sweep tone).")
        return SyntheticSource()
    return WasapiLoopbackSource()


def setup_display(cfg: Config, windowed: bool) -> tuple[int, int, int, int]:
    """Returns (grid_w, grid_h, screen_w, screen_h) and creates the pygame window."""
    pygame.display.init()
    if cfg.fullscreen and not windowed:
        info = pygame.display.Info()
        screen_w, screen_h = info.current_w, info.current_h
        pygame.display.set_mode((screen_w, screen_h), pygame.FULLSCREEN)
        grid_w = max(16, screen_w // cfg.pixel_size)
        grid_h = max(9, screen_h // cfg.pixel_size)
    else:
        grid_w, grid_h = cfg.grid_width, cfg.grid_height
        screen_w, screen_h = grid_w * cfg.window_scale, grid_h * cfg.window_scale
        pygame.display.set_mode((screen_w, screen_h))
    pygame.display.set_caption("Audio Visualizer")
    return grid_w, grid_h, screen_w, screen_h


def main() -> None:
    parser = argparse.ArgumentParser(description="Frequency-mapped halftone audio visualizer")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--demo", action="store_true", help="use a synthetic test tone instead of system audio")
    parser.add_argument("--windowed", action="store_true", help="override fullscreen for testing")
    args = parser.parse_args()

    cfg = Config.load(args.config)
    source = build_source(args.demo)

    pygame.init()
    grid_w, grid_h, screen_w, screen_h = setup_display(cfg, args.windowed)
    renderer = BitmapRenderer(grid_w, grid_h, screen_w, screen_h)

    analyzer = SpectrumAnalyzer(
        sample_rate=source.sample_rate,
        num_bands=cfg.num_freq_points,
        decay=cfg.decay,
        gain=cfg.gain,
    )
    cluster_w = max(8, grid_w // cfg.cluster_scale)
    cluster_h = max(8, grid_h // cfg.cluster_scale)
    spectral_field = SpectralField(cfg.num_freq_points, cluster_w, cluster_h, persistence=cfg.persistence)

    clock = pygame.time.Clock()

    now_playing = NowPlayingWatcher()
    now_playing.start()

    depth_idx = BIT_DEPTHS.index(cfg.bit_depth) if cfg.bit_depth in BIT_DEPTHS else 0
    mode_idx = MODES.index(cfg.mode) if cfg.mode in MODES else 0

    frame_iter = source.frames(CHUNK_SIZE)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_b:
                    depth_idx = (depth_idx + 1) % len(BIT_DEPTHS)
                    cfg.bit_depth = BIT_DEPTHS[depth_idx]
                elif event.key == pygame.K_m:
                    mode_idx = (mode_idx + 1) % len(MODES)
                    cfg.mode = MODES[mode_idx]
                elif event.key == pygame.K_n:
                    cfg.show_now_playing = not cfg.show_now_playing
                elif event.key == pygame.K_s:
                    cfg.save(args.config)
                    print(f"saved settings to {args.config}")

        samples = next(frame_iter)
        colors = grayscale_palette(cfg.bit_depth)
        band_levels = analyzer.process(samples)

        if cfg.mode == "field":
            cluster = spectral_field.update(band_levels)
            intensity = resize_nearest(cluster, grid_h, grid_w)

            left_text = right_text = ""
            if cfg.show_now_playing:
                label = now_playing.current().label()
                hint = f"{cfg.bit_depth}-BIT"
                if cfg.text_corner == "bottom_right":
                    left_text, right_text = hint, label
                else:
                    left_text, right_text = label, hint

            renderer.render_field(intensity, colors, left_text, right_text)
        else:
            mono = samples.mean(axis=1) if samples.ndim > 1 else samples
            renderer.render_waveform(mono, colors)

        clock.tick(cfg.fps)

    now_playing.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
