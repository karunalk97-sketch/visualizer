from __future__ import annotations

import argparse
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

# Keep the window up when focus moves elsewhere (another app, another monitor).
# Without this SDL minimizes a fullscreen window the moment it loses focus.
os.environ.setdefault("SDL_VIDEO_MINIMIZE_ON_FOCUS_LOSS", "0")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from .audio_router import AudioRouter  # noqa: E402
from .config import GLYPH_CELL_STEPS, PIXEL_STEPS, Config  # noqa: E402
from .features import FFT_SIZE, AudioFeatures  # noqa: E402
from .glyphs import GlyphField  # noqa: E402
from .now_playing import NowPlayingWatcher  # noqa: E402
from .palette import grayscale_palette  # noqa: E402
from .randomizer import randomize  # noqa: E402
from .renderer import BitmapRenderer, bar_height  # noqa: E402
from . import presets  # noqa: E402
from .engines import Fusion  # noqa: E402
from .settings_ui import SettingsPanel  # noqa: E402
from .window_style import blacken_title_bar  # noqa: E402

CHUNK_SIZE = FFT_SIZE                # the latest window of audio analysed every frame
BIT_DEPTHS = [1, 2, 3, 4]
MODES = ["field", "waveform"]
MAX_GRID_CELLS = 130_000            # keeps big screens at 60 fps: cells grow a little beyond this
MIN_WINDOW = (320, 200)
IDLE_SECONDS = 4                    # this long without sound and no track title: say so


class IntensitySmoother:
    """Blends this frame's picture with the last one, so the dithered pixels don't
    flicker ("glitter") as fast as the shapes change underneath -- a softer, rounder
    feel instead of a sharp one. `decay` 0 disables it (each frame stands on its own,
    today's default); higher values linger longer."""

    def __init__(self) -> None:
        self._buf: np.ndarray | None = None

    def apply(self, intensity: np.ndarray, decay: float) -> np.ndarray:
        if decay <= 0.0:
            self._buf = None
            return intensity
        if self._buf is None or self._buf.shape != intensity.shape:
            self._buf = intensity.astype(np.float32, copy=True)
        else:
            self._buf = self._buf * decay + intensity * (1.0 - decay)
        return self._buf


def idle_text(track_label: str, silent_frames: int, fps: int) -> str:
    """The status-bar title, or a hint when nothing has played for a while, so a
    black screen doesn't look like a broken app."""
    if track_label or silent_frames < fps * IDLE_SECONDS:
        return track_label
    return "Nothing is playing yet — start some music"


def crash_log_path() -> Path:
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_STATE_HOME") or tempfile.gettempdir()
    return Path(base) / "AudioVisualizer" / "error.log"


def report_crash(exc: BaseException, show_dialog: bool = True) -> Path:
    """The windowed app has no console, so an unexpected error would just make the
    window vanish. Write it to a log file (and tell the user where, on Windows)."""
    path = crash_log_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(f"--- {time.strftime('%Y-%m-%d %H:%M:%S')} ---\n")
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
            fh.write("\n")
    except OSError:
        pass
    if show_dialog and sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, f"Audio Visualizer hit an unexpected error and had to close.\n\n{type(exc).__name__}: {exc}\n\n"
                   f"Details were saved to:\n{path}", "Audio Visualizer", 0x10)
        except Exception:
            pass
    return path


def make_icon() -> pygame.Surface:
    icon = pygame.Surface((32, 32))
    icon.fill((0, 0, 0))
    for x, y, r in ((9, 10, 5), (21, 8, 3), (22, 21, 6), (8, 24, 2)):
        pygame.draw.circle(icon, (255, 255, 255), (x, y), r)
    return icon


def current_display() -> int:
    """Index of the monitor the window is on (0 if it can't be told)."""
    try:
        from pygame._sdl2.video import Window
        return int(Window.from_display_module().display_index)
    except Exception:
        return 0


def open_window(fullscreen: bool, size: tuple[int, int], display: int = 0) -> tuple[int, int]:
    """(Re)creates the window and returns its pixel size. Windowed mode is
    resizable with a (black) title bar carrying minimize / maximize / X. `display`
    keeps it on the same monitor when switching fullscreen on and off."""
    if display >= pygame.display.get_num_displays():
        display = 0
    if fullscreen:
        surface = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=display)
    else:
        surface = pygame.display.set_mode(size, pygame.RESIZABLE, display=display)
    pygame.display.set_caption("Audio Visualizer")
    blacken_title_bar()
    return surface.get_size()


def cell_size(cfg: Config, screen_h: int, screen_w: int = 1280) -> int:
    """Screen pixels per cell of the picture: dither pixels, or one character. On very
    large screens the pixel cells grow just enough to stay within the cell budget."""
    field_h = max(1, screen_h - bar_height(screen_h))
    if cfg.render_mode == "chars":
        return max(6, cfg.glyph_cell)
    fused = max(1, len(cfg.versions))
    budget = MAX_GRID_CELLS / (1.0 + 0.5 * (fused - 1))   # several versions at once: a little chunkier, still smooth
    return max(cfg.pixel_size, math.ceil(math.sqrt(screen_w * field_h / budget)))


def grid_for(cfg: Config, screen_w: int, screen_h: int) -> tuple[int, int]:
    """Cells across and down the picture area (the window minus the status bar)."""
    px = cell_size(cfg, screen_h, screen_w)
    field_h = max(1, screen_h - bar_height(screen_h))
    return max(4, screen_w // px), max(3, field_h // px)


def main(argv: list[str] | None = None) -> None:
    try:
        run(argv)
    except (SystemExit, KeyboardInterrupt):
        raise
    except Exception as exc:  # last resort: keep a record instead of vanishing silently
        report_crash(exc)
        try:
            pygame.quit()
        except Exception:
            pass
        raise SystemExit(1)


def run(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Black & white, 1-bit audio visualizer")
    parser.add_argument("--demo", action="store_true", help="use a synthetic test tone instead of system audio")
    parser.add_argument("--fullscreen", action="store_true", help="start fullscreen")
    parser.add_argument("--app", metavar="NAME", help="listen to one application only, e.g. --app spotify (Windows)")
    parser.add_argument("--preset", metavar="CODE_OR_NAME", help="start with a shared preset code (AVP1....) or a saved preset's name")
    parser.add_argument("--stats", metavar="FILE", help=argparse.SUPPRESS)   # dev: write frame-time stats on exit
    args = parser.parse_args(argv)

    cfg = Config()  # settings last for this session only; every launch starts fresh from the default preset
    presets.apply(cfg, presets.DEFAULT_PRESET)
    if args.preset:
        store = presets.PresetStore()
        saved = next((p["settings"] for p in store.load() if p["name"].lower() == args.preset.strip().lower()), None)
        try:
            presets.apply(cfg, saved if saved is not None else presets.decode(args.preset))
        except ValueError:
            pass                                          # not a code or a saved name: keep the default look
    cfg.fullscreen = args.fullscreen
    audio = AudioRouter(demo=args.demo)
    if args.app:
        name = args.app.strip().lower()
        audio.select("app:" + (name[:-4] if name.endswith(".exe") else name))

    pygame.init()
    pygame.display.set_icon(make_icon())
    fullscreen = cfg.fullscreen
    windowed_size = (cfg.window_width, cfg.window_height)
    screen_w, screen_h = open_window(fullscreen, windowed_size)
    grid_w, grid_h = grid_for(cfg, screen_w, screen_h)
    renderer = BitmapRenderer(grid_w, grid_h, screen_w, screen_h)

    features = AudioFeatures(audio.sample_rate, gain=cfg.gain)
    scene = Fusion(grid_w, grid_h, sample_rate=audio.sample_rate)     # every version, fused as chosen in settings
    glyph_field = GlyphField()
    smoother = IntensitySmoother()

    clock = pygame.time.Clock()
    now_playing = NowPlayingWatcher()
    now_playing.start()

    depth_idx = BIT_DEPTHS.index(cfg.bit_depth) if cfg.bit_depth in BIT_DEPTHS else 0
    mode_idx = MODES.index(cfg.mode) if cfg.mode in MODES else 0

    last_track: tuple[str, str] | None = None
    quiet_frames = 0
    silent_frames = 0

    def relayout(width: int, height: int) -> None:
        nonlocal screen_w, screen_h, grid_w, grid_h
        screen_w, screen_h = width, height
        grid_w, grid_h = grid_for(cfg, screen_w, screen_h)
        renderer.resize(grid_w, grid_h, screen_w, screen_h)
        scene.resize(grid_w, grid_h)

    def layout_signature() -> tuple:
        return (cfg.render_mode, cfg.pixel_size, cfg.glyph_cell, screen_w, screen_h, len(cfg.versions))

    layout_sig = layout_signature()

    display_idx = 0

    def set_fullscreen(on: bool) -> None:
        nonlocal fullscreen, display_idx
        if on != fullscreen:
            display_idx = current_display()               # stay on the monitor the window is on
        fullscreen = on
        cfg.fullscreen = on
        relayout(*open_window(fullscreen, windowed_size, display_idx))

    def reshuffle() -> None:
        """A new song (or R / Randomize): a new character, layout and terrain."""
        scene.reshuffle()
        features.reset_song()
        glyph_field.reshuffle()

    def do_randomize(within: bool = False) -> None:
        """Across every version (Space, the Randomize button), or only within the chosen ones."""
        nonlocal depth_idx
        randomize(cfg, within=within)
        depth_idx = BIT_DEPTHS.index(cfg.bit_depth)
        reshuffle()

    def on_setting_changed(name: str) -> None:
        nonlocal depth_idx
        if name == "fullscreen" and cfg.fullscreen != fullscreen:
            set_fullscreen(cfg.fullscreen)
        elif name in ("bit_depth", "preset") and cfg.bit_depth in BIT_DEPTHS:
            depth_idx = BIT_DEPTHS.index(cfg.bit_depth)

    preset_manager = presets.PresetManager(cfg, on_applied=lambda: on_setting_changed("preset"))
    panel = SettingsPanel(cfg, on_setting_changed, reshuffle, do_randomize, audio=audio,
                          on_randomize_within=lambda: do_randomize(within=True), presets=preset_manager)
    renderer.overlay = lambda surface: panel.draw(surface, screen_w, renderer.field_h)

    def step_size(direction: int) -> None:
        """direction -1 = tighter (more, smaller cells), +1 = looser (chunkier)."""
        if cfg.render_mode == "chars":
            steps, attr = GLYPH_CELL_STEPS, "glyph_cell"
        else:
            steps, attr = PIXEL_STEPS, "pixel_size"
        cur = getattr(cfg, attr)
        nearest = min(range(len(steps)), key=lambda i: abs(steps[i] - cur))
        setattr(cfg, attr, steps[max(0, min(len(steps) - 1, nearest + direction))])

    minimized = False
    running = True
    dt = 1.0 / cfg.fps
    frame_times: list[float] = []
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:  # the window's X button
                running = False
                continue
            if event.type == pygame.WINDOWMINIMIZED:
                minimized = True
            elif event.type in (pygame.WINDOWRESTORED, pygame.WINDOWMAXIMIZED, pygame.WINDOWSHOWN, pygame.WINDOWEXPOSED):
                minimized = False
            if event.type == pygame.VIDEORESIZE:
                if not fullscreen and event.w >= MIN_WINDOW[0] and event.h >= MIN_WINDOW[1]:
                    windowed_size = (event.w, event.h)
                    relayout(event.w, event.h)
                continue
            if panel.handle_event(event, screen_w, renderer.field_h):
                continue
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and event.pos[1] >= renderer.field_h:
                panel.toggle()  # a click on the status bar opens settings
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_f, pygame.K_F11):
                    set_fullscreen(not fullscreen)
                elif event.key == pygame.K_ESCAPE and fullscreen:
                    set_fullscreen(False)  # Esc only leaves fullscreen; it never closes the app
                elif event.key == pygame.K_SPACE:
                    do_randomize()
                elif event.key == pygame.K_r:
                    reshuffle()
                elif event.key == pygame.K_UP:
                    step_size(-1)
                elif event.key == pygame.K_DOWN:
                    step_size(+1)
                elif event.key == pygame.K_b:
                    depth_idx = (depth_idx + 1) % len(BIT_DEPTHS)
                    cfg.bit_depth = BIT_DEPTHS[depth_idx]
                elif event.key == pygame.K_c:
                    cfg.render_mode = "pixels" if cfg.render_mode == "chars" else "chars"
                elif event.key == pygame.K_m:
                    mode_idx = (mode_idx + 1) % len(MODES)
                    cfg.mode = MODES[mode_idx]
                elif event.key == pygame.K_n:
                    cfg.show_now_playing = not cfg.show_now_playing

        if minimized:                         # nothing to see: don't spend CPU drawing
            clock.tick(10)
            continue

        if layout_signature() != layout_sig:  # cell size, render mode... changed (keys or panel)
            layout_sig = layout_signature()
            relayout(screen_w, screen_h)

        # live settings
        features.gain = cfg.gain
        scene.configure(cfg.versions, cfg.elements,
                        mix=dict(bass=cfg.mix_bass, lowmid=cfg.mix_lowmid, vocals=cfg.mix_vocals,
                                 highmid=cfg.mix_highmid, treble=cfg.mix_treble),
                        intensity=cfg.intensity, invert=cfg.overlap_invert)

        samples = audio.read(CHUNK_SIZE)                  # never blocks, never raises: silence when nothing is playing
        features.set_sample_rate(audio.sample_rate)         # the source (device or app) may have changed
        now_playing.prefer(audio.wanted_key)                # show the title of the app we are actually listening to
        colors = grayscale_palette(cfg.bit_depth)
        feat = features.update(samples, dt)
        if feat.section_change:
            scene.new_section(feat.section)
        silent_frames = silent_frames + 1 if feat.silent else 0

        # New song -> new random layout. Detected from the OS now-playing info,
        # or (when that's unavailable) from music resuming after a gap.
        if cfg.reshuffle_on_new_song:
            track = now_playing.current()
            key = (track.title, track.artist)
            if key != ("", ""):
                if last_track is not None and key != last_track:
                    reshuffle()
                last_track = key
            if feat.silent:
                quiet_frames += 1
            else:
                if quiet_frames > cfg.fps * 2 and last_track is None:
                    reshuffle()
                quiet_frames = 0

        left_text = right_text = ""
        if cfg.show_now_playing:
            label = idle_text(now_playing.current().label(), silent_frames, cfg.fps)
            px = cell_size(cfg, screen_h, screen_w)
            what = f"{px}px chars" if cfg.render_mode == "chars" else f"{px}px"
            hint = f"{audio.label}   {chr(0xB7)}   Tab: settings   {chr(0xB7)}   {cfg.bit_depth}-bit  {chr(0xB7)}  {what}"
            if cfg.text_corner == "bottom_right":
                left_text, right_text = hint, label
            else:
                left_text, right_text = label, hint

        if cfg.mode == "field":
            intensity = smoother.apply(scene.update(samples, audio.sample_rate, feat, dt), cfg.pixel_decay)
            if cfg.render_mode == "chars":
                glyph_field.configure(cfg.glyph_sets, max(6, cfg.glyph_cell), cfg.bit_depth)
                renderer.render_gray(glyph_field.render(intensity, cfg.glyph_mapping), left_text, right_text)
            else:
                renderer.render_field(intensity, colors, left_text, right_text)
        else:
            mono = samples.mean(axis=1) if samples.ndim > 1 else samples
            renderer.render_waveform(mono[-grid_w * 4:], colors, left_text, right_text)

        dt = clock.tick(cfg.fps) / 1000.0                   # real frame time: motion is the same at any frame rate
        if args.stats:
            frame_times.append(dt)

    if args.stats and frame_times:
        ft = sorted(frame_times[60:] or frame_times)
        Path(args.stats).write_text(
            f"frames {len(frame_times)}  mean_fps {len(ft) / sum(ft):.1f}  "
            f"p95_ms {1000 * ft[int(len(ft) * 0.95) - 1]:.1f}  grid {grid_w}x{grid_h}\n")
    audio.close()
    now_playing.stop()
    pygame.quit()


if __name__ == "__main__":
    main()
