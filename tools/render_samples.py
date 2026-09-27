"""Renders headless screenshots of the visualizer for different kinds of audio,
running the same pipeline as the app (shapes + relief + pixels
or characters + status bar + optional settings panel).

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
from visualizer.main import grid_for, shape_grid  # noqa: E402
from visualizer.palette import grayscale_palette  # noqa: E402
from visualizer.randomizer import randomize  # noqa: E402
from visualizer.relief import relief  # noqa: E402
from visualizer.renderer import BitmapRenderer  # noqa: E402
from visualizer.settings_ui import SettingsPanel  # noqa: E402
from visualizer.spectral_field import SpectralField, resize_bilinear  # noqa: E402

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


def hard_kick(seconds: float) -> np.ndarray:
    """A hard bass hit with the broadband click that spreads across the spectrum."""
    t = _t(seconds)
    out = np.zeros_like(t)
    for s in np.arange(0, seconds, 0.5):
        i, n = int(s * SR), int(0.25 * SR)
        tt = np.arange(n) / SR
        out[i:i + n] += np.sin(2 * np.pi * (45 + 80 * np.exp(-tt * 25)) * tt) * np.exp(-tt * 9) * 0.9
        out[i:i + int(0.004 * SR)] += np.random.default_rng(1).standard_normal(int(0.004 * SR)) * 0.5
    return out.astype(np.float32)


def hi_hat(seconds: float) -> np.ndarray:
    t = _t(seconds)
    out = np.zeros_like(t)
    rng = np.random.default_rng(2)
    for s in np.arange(0.25, seconds, 0.5):
        i, n = int(s * SR), int(0.06 * SR)
        tt = np.arange(n) / SR
        out[i:i + n] += np.diff(rng.standard_normal(n + 1)) * np.exp(-tt * 80) * 0.5
    return out.astype(np.float32)


STYLES = {
    "bass_heavy": bass_heavy,
    "treble_bright": treble_bright,
    "vocal": vocal,
    "synth_pad": synth_pad,
    "pad_and_drums": pad_and_drums,
    "full_mix": full_mix,
}


def render_frames(chunks, seed: int, out_path: Path, warmup: int = 240, cfg: Config | None = None,
                  label: str = "Midnight Drive - The Sample Band", crop: tuple | None = None,
                  panel: bool = False, tab: str = "look", pick_peak: bool = False,
                  **cfg_overrides) -> np.ndarray:
    """Runs the audio through the app's pipeline and saves one frame. With
    `pick_peak` it keeps the brightest frame after warmup (the hardest hit)."""
    cfg = dataclasses.replace(cfg or Config(), **cfg_overrides)
    pygame.display.set_mode((W, H))
    grid_w, grid_h = grid_for(cfg, W, H)
    renderer = BitmapRenderer(grid_w, grid_h, W, H)
    analyzer = SpectrumAnalyzer(sample_rate=SR, num_bands=96)
    field = SpectralField(96, *shape_grid(W, H), seed=seed, invert=cfg.overlap_invert)
    glyphs = GlyphField(seed=seed)
    ui = SettingsPanel(cfg, lambda name: None, lambda: None, lambda: None)
    ui.visible, ui.tab = panel, tab
    renderer.overlay = lambda surface: ui.draw(surface, W, renderer.field_h)

    best = (-1.0, None)
    levels = None
    n = 0
    for chunk in chunks:
        levels = analyzer.process(chunk)
        base = field.update(levels)
        n += 1
        if n < warmup:
            continue
        intensity = relief(resize_bilinear(base, grid_h, grid_w), cfg.depth)
        if pick_peak:
            score = float(base.mean())
            if score > best[0]:
                best = (score, intensity.copy())
            continue
        best = (0.0, intensity)
        break
    intensity = best[1] if best[1] is not None else np.zeros((grid_h, grid_w), np.float32)
    hint = f"Tab: settings   {chr(0xB7)}   {cfg.bit_depth}-bit  {chr(0xB7)}  "
    if cfg.render_mode == "chars":
        glyphs.configure(cfg.glyph_sets, cfg.glyph_cell, cfg.bit_depth)
        renderer.render_gray(glyphs.render(intensity, cfg.glyph_mapping), label, hint + f"{cfg.glyph_cell}px chars")
    else:
        renderer.render_field(intensity, grayscale_palette(cfg.bit_depth), label, hint + f"{cfg.pixel_size}px")
    surface = pygame.display.get_surface()
    if crop:
        surface = surface.subsurface(pygame.Rect(crop))
    pygame.image.save(surface, str(out_path))
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
        lv = render_frames(chunked(fn(14.0)), seed=11, out_path=out / f"{name}.png")
        print(f"{name}: mean level {lv.mean():.2f}, peak {lv.max():.2f}")

    mix = lambda: chunked(pad_and_drums(14.0))                # noqa: E731
    # the worst moment of a hard kick vs a hi-hat: neither may fill the screen
    render_frames(chunked(hard_kick(8.0)), 11, out / "hit_kick.png", pick_peak=True, warmup=60)
    render_frames(chunked(hi_hat(8.0)), 11, out / "hit_hat.png", pick_peak=True, warmup=60)
    # 3D depth relief off vs on (same frame)
    render_frames(mix(), 11, out / "depth_off.png", depth=0.0)
    render_frames(mix(), 11, out / "depth_on.png", depth=0.6)
    # character sets
    base = dict(render_mode="chars", glyph_cell=12)
    render_frames(mix(), 11, out / "chars_shapes.png", glyph_sets=["shapes"], **base)
    render_frames(mix(), 11, out / "chars_symbols.png", glyph_sets=["symbols"], **base)
    render_frames(mix(), 11, out / "chars_ascii.png", glyph_sets=["ascii"], glyph_mapping="brightness", **{**base, "glyph_cell": 10})
    render_frames(mix(), 11, out / "chars_binary.png", glyph_sets=["binary"], **{**base, "glyph_cell": 10})
    render_frames(mix(), 11, out / "chars_mixed.png", glyph_sets=["shapes", "symbols", "ascii"], **base)
    # the settings panel, both tabs
    render_frames(mix(), 11, out / "panel_look.png", panel=True, tab="look")
    render_frames(mix(), 11, out / "panel_chars.png", panel=True, tab="chars", render_mode="chars", glyph_sets=["shapes", "symbols"])
    # what the Randomize button gives you
    rng = np.random.default_rng(21)
    for i in range(6):
        cfg = Config()
        randomize(cfg, rng)
        render_frames(mix(), 11 + i, out / f"random_{i}.png", cfg=cfg)

    if args.live:
        print(f"capturing {args.live}s of system audio ...")
        lv = render_frames(capture_live(args.live), seed=5, out_path=out / "live_system_audio.png",
                           warmup=int(args.live * SR / CHUNK) - 2, label="Live capture - Spotify")
        print(f"live: mean level {lv.mean():.2f}, peak {lv.max():.2f}")

if __name__ == "__main__":
    main()
