"""Renders headless screenshots of the visualizer for different kinds of audio,
running the same pipeline as the app (shapes + ribbon waves + pixels or
characters + status bar + optional settings panel).

    python tools/render_samples.py out_dir            # synthetic music styles
    python tools/render_samples.py out_dir --live 20  # + real system audio (Windows)

Handy for checking layout/sensitivity changes without opening a window.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from visualizer.analyzer import SpectrumAnalyzer  # noqa: E402
from visualizer.config import Config  # noqa: E402
from visualizer.glyphs import GlyphField  # noqa: E402
from visualizer.main import DEFAULT_GLYPH_FONT, grid_for, shape_grid, wave_grid  # noqa: E402
from visualizer.palette import grayscale_palette  # noqa: E402
from visualizer.renderer import BitmapRenderer  # noqa: E402
from visualizer.settings_ui import SettingsPanel  # noqa: E402
from visualizer.spectral_field import SpectralField, resize_bilinear  # noqa: E402
from visualizer.waves import WaveField, compose  # noqa: E402

SR = 48000
CHUNK = 1024
W, H = 1280, 720


def _t(seconds: float) -> np.ndarray:
    return np.arange(int(seconds * SR)) / SR


def bass_heavy(seconds: float) -> np.ndarray:
    t = _t(seconds)
    beat = (t * 2.0) % 1.0  # kick at 120 bpm
    kick = np.sin(2 * np.pi * (45 + 40 * np.exp(-beat * 20)) * beat) * np.exp(-beat * 7)
    sub = 0.35 * np.sin(2 * np.pi * 41.2 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t))
    growl = 0.18 * np.sin(2 * np.pi * 82.4 * t + 2 * np.sin(2 * np.pi * 3 * t))
    return (0.7 * kick + sub + growl).astype(np.float32)


def treble_bright(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(1)
    t = _t(seconds)
    hat_env = np.exp(-((t * 8.0) % 1.0) * 14)
    hats = np.diff(rng.standard_normal(len(t) + 1)) * hat_env * 0.35
    step = (t * 6).astype(int) % 8
    notes = np.array([3520, 4186, 5274, 6272, 7040, 5274, 4186, 8372])[step]
    arp = 0.25 * np.sin(2 * np.pi * np.cumsum(notes) / SR)
    shimmer = 0.12 * np.sin(2 * np.pi * 9800 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.7 * t))
    return (hats + arp + shimmer).astype(np.float32)


def vocal(seconds: float) -> np.ndarray:
    t = _t(seconds)
    f0 = 200 + 40 * np.sin(2 * np.pi * 0.4 * t) + 6 * np.sin(2 * np.pi * 5.5 * t)
    phase = 2 * np.pi * np.cumsum(f0) / SR
    vowel = (t * 0.6) % 3  # slide between three vowel shapes
    formants = np.array([[730, 1090, 2440], [270, 2290, 3010], [300, 870, 2240]])
    out = np.zeros_like(t)
    for h in range(1, 30):
        hz = f0 * h
        weight = np.zeros_like(t)
        for k in range(3):
            fk = np.interp(vowel, [0, 1, 2, 3], np.append(formants[:, k], formants[0, k]))
            weight += np.exp(-((hz - fk) / (200 + 60 * k)) ** 2) / (1 + 0.4 * k)
        out += weight * np.sin(h * phase) / h ** 0.35
    return (0.35 * out / max(1e-6, np.abs(out).max()) * 2).astype(np.float32)


def synth_pad(seconds: float) -> np.ndarray:
    """Sustained detuned saw chords (Am7 -> Fmaj7 -> C -> G) with a gliding lead on top."""
    t = _t(seconds)
    chords = [(220.0, 261.6, 329.6, 392.0), (174.6, 220.0, 261.6, 329.6),
              (196.0, 261.6, 329.6, 392.0), (196.0, 246.9, 293.7, 392.0)]
    out = np.zeros_like(t)
    seg = ((t / 3.0).astype(int)) % len(chords)
    for i, chord in enumerate(chords):
        mask = (seg == i).astype(np.float32)
        mask = np.convolve(mask, np.ones(4800) / 4800, mode="same")  # soft chord changes
        for f in chord:
            for det in (0.997, 1.0, 1.004):
                for h in range(1, 9):
                    out += mask * np.sin(2 * np.pi * f * det * h * t) / h ** 1.2 * 0.05
    lead_f = 660 + 260 * np.sin(2 * np.pi * 0.23 * t)
    lead = 0.12 * np.sin(2 * np.pi * np.cumsum(lead_f) / SR) * (0.7 + 0.3 * np.sin(2 * np.pi * 5 * t))
    lead += 0.05 * np.sin(2 * np.pi * np.cumsum(lead_f * 2) / SR)
    return (out + lead).astype(np.float32)


def full_mix(seconds: float, root: float = 220.0) -> np.ndarray:
    t = _t(seconds)
    chord = sum(np.sin(2 * np.pi * root * r * t) + 0.4 * np.sin(2 * np.pi * root * r * 2 * t)
                for r in (1.0, 1.2, 1.5, 1.8))
    chord = 0.12 * chord * (0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    return (0.55 * bass_heavy(seconds)[: len(t)] + chord + 0.5 * treble_bright(seconds)[: len(t)]
            + 0.6 * vocal(seconds)[: len(t)]).astype(np.float32)


def pad_and_drums(seconds: float) -> np.ndarray:
    return (synth_pad(seconds) + 0.9 * bass_heavy(seconds)).astype(np.float32)


def drums_only(seconds: float) -> np.ndarray:
    rng = np.random.default_rng(3)
    t = _t(seconds)
    hat = np.diff(rng.standard_normal(len(t) + 1)) * np.exp(-((t * 4.0) % 1.0) * 18) * 0.4
    return (0.8 * bass_heavy(seconds) + hat).astype(np.float32)


STYLES = {
    "bass_heavy": bass_heavy,
    "treble_bright": treble_bright,
    "vocal": vocal,
    "synth_pad": synth_pad,
    "pad_and_drums": pad_and_drums,
    "drums_only": drums_only,
    "full_mix": full_mix,
}


def render_frames(chunks, seed: int, out_path: Path, warmup: int = 240, cfg: Config | None = None,
                  label: str = "Midnight Drive - The Sample Band", crop: tuple | None = None,
                  panel: bool = False, want_ribbons: int = 0, **cfg_overrides) -> np.ndarray:
    """Runs the audio through the app's pipeline and saves the frame after `warmup`
    chunks (continuing until `want_ribbons` ribbons are on screen, if asked)."""
    cfg = dataclasses.replace(cfg or Config(), **cfg_overrides)
    pygame.display.set_mode((W, H))
    grid_w, grid_h = grid_for(cfg, W, H)
    renderer = BitmapRenderer(grid_w, grid_h, W, H)
    analyzer = SpectrumAnalyzer(sample_rate=SR, num_bands=96)
    field = SpectralField(96, *shape_grid(W, H), seed=seed, invert=cfg.overlap_invert)
    waves = WaveField(96, *wave_grid(W, H), seed=seed, strength=cfg.wave_strength,
                      softness=cfg.wave_softness, rate=cfg.wave_rate)
    glyphs = GlyphField(seed=seed)
    ui = SettingsPanel(cfg, lambda name: None, lambda: None)
    ui.visible = panel
    renderer.overlay = lambda surface: ui.draw(surface, W, renderer.field_h)

    levels = intensity = None
    it = iter(chunks)
    n = 0
    while True:
        chunk = next(it, None)
        if chunk is None:
            break
        levels = analyzer.process(chunk)
        base = field.update(levels)
        wv = waves.update(levels) if cfg.waves else None
        n += 1
        done = n >= warmup and (waves.ribbon_count >= want_ribbons or n > warmup + 900)
        if done:
            intensity = (resize_bilinear(base, grid_h, grid_w) if cfg.show_shapes
                         else np.zeros((grid_h, grid_w), np.float32))
            if wv is not None and waves.active:
                intensity = compose(intensity, resize_bilinear(wv, grid_h, grid_w))
            break
    if intensity is None:
        intensity = resize_bilinear(base, grid_h, grid_w) if cfg.show_shapes else np.zeros((grid_h, grid_w), np.float32)
    hint = f"Tab: settings   {chr(0xB7)}   {cfg.bit_depth}-bit  {chr(0xB7)}  "
    if cfg.render_mode == "chars":
        glyphs.configure(cfg.glyph_shapes, cfg.glyph_chars, cfg.glyph_font or DEFAULT_GLYPH_FONT, cfg.glyph_cell, cfg.bit_depth)
        renderer.render_gray(glyphs.render(intensity, cfg.glyph_mapping), label, hint + f"{cfg.glyph_cell}px chars")
    else:
        renderer.render_field(intensity, grayscale_palette(cfg.bit_depth), label, hint + f"{cfg.pixel_size}px")
    surf = pygame.display.get_surface()
    if crop:
        surf = surf.subsurface(pygame.Rect(crop))
    pygame.image.save(surf, str(out_path))
    return levels


def chunked(signal: np.ndarray):
    for i in range(0, len(signal) - CHUNK, CHUNK):
        yield signal[i:i + CHUNK]


def capture_live(seconds: float):
    from visualizer.audio_capture import WasapiLoopbackSource
    src = WasapiLoopbackSource()
    global SR
    SR = src.sample_rate
    frames = src.frames(CHUNK)
    n = int(seconds * SR / CHUNK)
    for _ in range(n):
        yield next(frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out_dir")
    parser.add_argument("--live", type=float, default=0, help="also capture N seconds of real system audio")
    args = parser.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pygame.init()

    for name, fn in STYLES.items():
        lv = render_frames(chunked(fn(12.0)), seed=11, out_path=out / f"{name}.png", want_ribbons=1 if name in ("synth_pad", "vocal") else 0)
        print(f"{name}: mean level {lv.mean():.2f}, peak {lv.max():.2f}")

    pad = lambda: chunked(synth_pad(12.0))          # noqa: E731
    mix = lambda: chunked(pad_and_drums(12.0))      # noqa: E731
    # layers on / off
    render_frames(pad(), 11, out / "layers_both.png", want_ribbons=1)
    render_frames(pad(), 11, out / "layers_waves_only.png", want_ribbons=1, show_shapes=False)
    render_frames(pad(), 11, out / "layers_shapes_only.png", want_ribbons=1, waves=False)
    # character mode
    chars = dict(render_mode="chars", want_ribbons=1)
    render_frames(mix(), 11, out / "chars_shapes.png", **chars)
    render_frames(mix(), 11, out / "chars_wingdings.png", glyph_shapes=[], glyph_chars="lmnopqrsuvxy", glyph_font="wingdings", **chars)
    render_frames(mix(), 11, out / "chars_text.png", glyph_shapes=[], glyph_chars="01", glyph_cell=10, **chars)
    render_frames(mix(), 11, out / "chars_bright.png", glyph_shapes=[], glyph_chars="·:+*#@", glyph_mapping="brightness",
                  glyph_cell=10, glyph_font="consolas", **chars)
    render_frames(mix(), 11, out / "chars_blocks.png", glyph_shapes=[], glyph_chars="░▒▓█", glyph_mapping="brightness",
                  glyph_cell=10, bit_depth=3, **chars)
    # the settings panel
    render_frames(mix(), 11, out / "panel_pixels.png", panel=True)
    render_frames(mix(), 11, out / "panel_chars.png", panel=True, glyph_chars="lmnop", glyph_font="wingdings", **chars)

    quiet = np.concatenate([synth_pad(3.0), np.zeros(SR * 4, dtype=np.float32)])
    render_frames(chunked(quiet), seed=11, out_path=out / "after_silence.png", warmup=10 ** 6)

    if args.live:
        print(f"capturing {args.live}s of system audio ...")
        lv = render_frames(capture_live(args.live), seed=5, out_path=out / "live_system_audio.png",
                           warmup=int(args.live * SR / CHUNK) - 2, label="Live capture - Spotify")
        print(f"live: mean level {lv.mean():.2f}, peak {lv.max():.2f}")


if __name__ == "__main__":
    main()
