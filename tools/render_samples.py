"""Renders headless screenshots of the visualizer for different kinds of audio.

    python tools/render_samples.py out_dir            # synthetic music styles
    python tools/render_samples.py out_dir --live 20  # + real system audio (Windows)

Handy for checking layout/sensitivity changes without opening a window.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from visualizer.analyzer import SpectrumAnalyzer  # noqa: E402
from visualizer.palette import grayscale_palette  # noqa: E402
from visualizer.renderer import BitmapRenderer  # noqa: E402
from visualizer.spectral_field import SpectralField, resize_nearest  # noqa: E402

SR = 48000
CHUNK = 1024
W, H, PIXEL = 1280, 720, 6


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


def full_mix(seconds: float, root: float = 220.0) -> np.ndarray:
    t = _t(seconds)
    chord = sum(np.sin(2 * np.pi * root * r * t) + 0.4 * np.sin(2 * np.pi * root * r * 2 * t)
                for r in (1.0, 1.2, 1.5, 1.8))
    chord = 0.12 * chord * (0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t))
    return (0.55 * bass_heavy(seconds)[: len(t)] + chord + 0.5 * treble_bright(seconds)[: len(t)]
            + 0.6 * vocal(seconds)[: len(t)]).astype(np.float32)


STYLES = {
    "bass_heavy": bass_heavy,
    "treble_bright": treble_bright,
    "vocal": vocal,
    "full_mix": full_mix,
}


def render_frames(chunks, seed: int, out_path: Path, warmup: int = 140) -> np.ndarray:
    surf = pygame.display.set_mode((W, H))
    gw, gh = W // PIXEL, H // PIXEL
    renderer = BitmapRenderer(gw, gh, W, H)
    analyzer = SpectrumAnalyzer(sample_rate=SR, num_bands=96)
    field = SpectralField(96, gw // 3, gh // 3, seed=seed)
    colors = grayscale_palette(1)
    levels = None
    for n, chunk in enumerate(chunks):
        levels = analyzer.process(chunk)
        cluster = field.update(levels)
        if n >= warmup:
            break
    renderer.render_field(resize_nearest(cluster, gh, gw), colors, "", "")
    pygame.image.save(pygame.display.get_surface(), str(out_path))
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
        lv = render_frames(chunked(fn(4.0)), seed=11, out_path=out / f"{name}.png")
        print(f"{name}: mean level {lv.mean():.2f}, peak {lv.max():.2f}")

    # same song, two different "songs" worth of layout
    mix = full_mix(4.0)
    render_frames(chunked(mix), seed=101, out_path=out / "full_mix_layout_A.png")
    render_frames(chunked(mix), seed=202, out_path=out / "full_mix_layout_B.png")

    if args.live:
        print(f"capturing {args.live}s of system audio ...")
        lv = render_frames(capture_live(args.live), seed=5, out_path=out / "live_system_audio.png",
                           warmup=int(args.live * SR / CHUNK) - 2)
        print(f"live: mean level {lv.mean():.2f}, peak {lv.max():.2f}")


if __name__ == "__main__":
    main()
