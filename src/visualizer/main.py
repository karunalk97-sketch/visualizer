from __future__ import annotations

import argparse
import math
import os
import sys

# Keep the window up when focus moves elsewhere (another app, another monitor).
# Without this SDL minimizes a fullscreen window the moment it loses focus.
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")

import pygame  # noqa: E402

from .analyzer import SpectrumAnalyzer  # noqa: E402
from .audio_capture import InputDeviceSource, SyntheticSource, WasapiLoopbackSource  # noqa: E402
from .config import Config, default_config_path  # noqa: E402
from .now_playing import NowPlayingWatcher  # noqa: E402
from .palette import grayscale_palette  # noqa: E402
from .renderer import BitmapRenderer, bar_height  # noqa: E402
from .spectral_field import SpectralField, resize_bilinear  # noqa: E402
from .waves import WaveField, compose  # noqa: E402
from .window_style import blacken_title_bar  # noqa: E402

CHUNK_SIZE = 1024
BIT_DEPTHS = [1, 2, 3, 4]
MODES = ["field", "waveform"]
PIXEL_STEPS = [2, 3, 4, 6, 8, 12]   # screen pixels per bitmap cell, tightest first
MAX_GRID_ROWS = 400                 # keeps very large screens fast: coarsen the grid beyond this
MIN_WINDOW = (320, 200)


def build_source(demo: bool):
    if demo:
        return SyntheticSource()
    try:
        if sys.platform == "win32":
            return WasapiLoopbackSource()
        return InputDeviceSource()
    except Exception as exc:  # missing driver / no device: still open, with the demo tone
        print(f"Could not open system audio ({exc}); falling back to the demo tone.")
        return SyntheticSource()


def make_icon() -> pygame.Surface:
    icon = pygame.Surface((32, 32))
    icon.fill((0, 0, 0))
    for x, y, r in ((9, 10, 5), (21, 8, 3), (22, 21, 6), (8, 24, 2)):
        pygame.draw.circle(icon, (255, 255, 255), (x, y), r)
    return icon


def open_window(cfg: Config, fullscreen: bool, size: tuple[int, int]) -> tuple[int, int]:
    """(Re)creates the window and returns its pixel size. Windowed mode is
    resizable with a (black) title bar carrying minimize / maximize / X."""
    if fullscreen:
        surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
    else:
        surface = pygame.display.set_mode(size, pygame.RESIZABLE)
    pygame.display.set_caption("Audio Visualizer")
    blacken_title_bar()
    return surface.get_size()


def effective_pixel_size(cfg: Config, screen_h: int) -> int:
    field_h = max(1, screen_h - bar_height(screen_h))
    return max(cfg.pixel_size, math.ceil(field_h / MAX_GRID_ROWS))


def grid_for(cfg: Config, screen_w: int, screen_h: int) -> tuple[int, int]:
    """Bitmap size for the picture area (the window minus the status bar)."""
    px = effective_pixel_size(cfg, screen_h)
    field_h = max(1, screen_h - bar_height(screen_h))
    return max(16, screen_w // px), max(9, field_h // px)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Random-spot frequency audio visualizer")
    parser.add_argument("--config", default=None, help="settings file (default: per-user app data folder)")
    parser.add_argument("--demo", action="store_true", help="use a synthetic test tone instead of system audio")
    parser.add_argument("--windowed", action="store_true", help="start windowed even if settings say fullscreen")
    args = parser.parse_args(argv)

    config_path = args.config or default_config_path()
    cfg = Config.load(config_path)
    source = build_source(args.demo)

    pygame.init()
    pygame.display.set_icon(make_icon())
    fullscreen = cfg.fullscreen and not args.windowed
    windowed_size = (cfg.window_width, cfg.window_height)
    screen_w, screen_h = open_window(cfg, fullscreen, windowed_size)
    grid_w, grid_h = grid_for(cfg, screen_w, screen_h)
    renderer = BitmapRenderer(grid_w, grid_h, screen_w, screen_h)

    analyzer = SpectrumAnalyzer(
        sample_rate=source.sample_rate,
        num_bands=cfg.num_freq_points,
        decay=cfg.decay,
        gain=cfg.gain,
    )

    def cluster_size() -> tuple[int, int]:
        return max(8, grid_w // cfg.cluster_scale), max(8, grid_h // cfg.cluster_scale)

    def wave_size() -> tuple[int, int]:
        return max(16, grid_w // 2), max(9, grid_h // 2)  # finer than the spots so thin waves stay smooth

    cw, ch = cluster_size()
    spectral_field = SpectralField(
        cfg.num_freq_points, cw, ch,
        persistence=cfg.persistence,
        spots_per_bin=cfg.spots_per_bin,
        drift=cfg.drift,
        invert=cfg.overlap_invert,
    )
    wave_field = WaveField(cfg.num_freq_points, *wave_size(), strength=cfg.wave_strength)

    clock = pygame.time.Clock()
    now_playing = NowPlayingWatcher()
    now_playing.start()

    depth_idx = BIT_DEPTHS.index(cfg.bit_depth) if cfg.bit_depth in BIT_DEPTHS else 0
    mode_idx = MODES.index(cfg.mode) if cfg.mode in MODES else 0

    last_track: tuple[str, str] | None = None
    quiet_frames = 0

    def relayout(width: int, height: int) -> None:
        nonlocal screen_w, screen_h, grid_w, grid_h
        screen_w, screen_h = width, height
        grid_w, grid_h = grid_for(cfg, screen_w, screen_h)
        renderer.resize(grid_w, grid_h, screen_w, screen_h)
        cw, ch = cluster_size()
        spectral_field.resize(cw, ch)
        wave_field.resize(*wave_size())

    def set_fullscreen(on: bool) -> None:
        nonlocal fullscreen
        fullscreen = on
        relayout(*open_window(cfg, fullscreen, windowed_size))

    def reshuffle() -> None:
        spectral_field.reshuffle()
        wave_field.reshuffle()

    def step_pixel_size(direction: int) -> None:
        """direction -1 = tighter (more, smaller pixels), +1 = looser (chunkier)."""
        nearest = min(range(len(PIXEL_STEPS)), key=lambda i: abs(PIXEL_STEPS[i] - cfg.pixel_size))
        cfg.pixel_size = PIXEL_STEPS[max(0, min(len(PIXEL_STEPS) - 1, nearest + direction))]
        relayout(screen_w, screen_h)

    frame_iter = source.frames(CHUNK_SIZE)
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # the window's X button
                running = False
            elif event.type == pygame.VIDEORESIZE:
                if not fullscreen and event.w >= MIN_WINDOW[0] and event.h >= MIN_WINDOW[1]:
                    windowed_size = (event.w, event.h)
                    relayout(event.w, event.h)
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_f, pygame.K_F11):
                    set_fullscreen(not fullscreen)
                elif event.key == pygame.K_ESCAPE and fullscreen:
                    set_fullscreen(False)  # Esc only leaves fullscreen; it never closes the app
                elif event.key == pygame.K_r:
                    reshuffle()
                elif event.key == pygame.K_UP:
                    step_pixel_size(-1)
                elif event.key == pygame.K_DOWN:
                    step_pixel_size(+1)
                elif event.key == pygame.K_b:
                    depth_idx = (depth_idx + 1) % len(BIT_DEPTHS)
                    cfg.bit_depth = BIT_DEPTHS[depth_idx]
                elif event.key == pygame.K_w:
                    cfg.waves = not cfg.waves
                elif event.key == pygame.K_m:
                    mode_idx = (mode_idx + 1) % len(MODES)
                    cfg.mode = MODES[mode_idx]
                elif event.key == pygame.K_n:
                    cfg.show_now_playing = not cfg.show_now_playing
                elif event.key == pygame.K_s:
                    cfg.window_width, cfg.window_height = windowed_size
                    cfg.fullscreen = fullscreen
                    cfg.save(config_path)
                    print(f"saved settings to {config_path}")

        samples = next(frame_iter)
        colors = grayscale_palette(cfg.bit_depth)
        band_levels = analyzer.process(samples)

        # New song -> new random layout. Detected from the OS now-playing info,
        # or (when that's unavailable) from music resuming after a gap.
        if cfg.reshuffle_on_new_song:
            track = now_playing.current()
            key = (track.title, track.artist)
            if key != ("", ""):
                if last_track is not None and key != last_track:
                    reshuffle()
                last_track = key
            if float(band_levels.mean()) < 0.02:
                quiet_frames += 1
            else:
                if quiet_frames > cfg.fps * 2 and last_track is None:
                    reshuffle()
                quiet_frames = 0

        left_text = right_text = ""
        if cfg.show_now_playing:
            label = now_playing.current().label()
            hint = f"{cfg.bit_depth}-bit  {chr(0xB7)}  {effective_pixel_size(cfg, screen_h)}px"
            if cfg.text_corner == "bottom_right":
                left_text, right_text = hint, label
            else:
                left_text, right_text = label, hint

        if cfg.mode == "field":
            intensity = resize_bilinear(spectral_field.update(band_levels), grid_h, grid_w)
            wave_layer = wave_field.update(band_levels)
            if cfg.waves and wave_field.active:
                intensity = compose(intensity, resize_bilinear(wave_layer, grid_h, grid_w))
            renderer.render_field(intensity, colors, left_text, right_text)
        else:
            mono = samples.mean(axis=1) if samples.ndim > 1 else samples
            renderer.render_waveform(mono, colors, left_text, right_text)

        clock.tick(cfg.fps)

    now_playing.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
