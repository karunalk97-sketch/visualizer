"""Headless screenshots from *synthetic* music (no audio device needed), through the
app's real pipeline (features -> scene -> dither -> status bar). For real music use
tools/record_audio.py + tools/render_live.py instead -- that is what the look was
tuned on.

    python tools/render_samples.py out_dir
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from render_live import run  # noqa: E402
from visualizer.config import Config  # noqa: E402

SR = 48000


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
    return (hats + arp).astype(np.float32)


def synth_pad(seconds: float) -> np.ndarray:
    """Sustained detuned chords with a gliding lead on top."""
    t = _t(seconds)
    chords = [(220.0, 261.6, 329.6, 392.0), (174.6, 220.0, 261.6, 329.6),
              (196.0, 261.6, 329.6, 392.0), (196.0, 246.9, 293.7, 392.0)]
    out = np.zeros_like(t)
    seg = ((t / 3.0).astype(int)) % len(chords)
    for i, chord in enumerate(chords):
        mask = np.convolve((seg == i).astype(np.float32), np.ones(4800) / 4800, mode="same")
        for f in chord:
            for h in range(1, 6):
                out += mask * np.sin(2 * np.pi * f * h * t) / h ** 1.2 * 0.06
    lead_f = 660 + 260 * np.sin(2 * np.pi * 0.23 * t)
    out += 0.12 * np.sin(2 * np.pi * np.cumsum(lead_f) / SR)
    return out.astype(np.float32)


def pad_and_drums(seconds: float) -> np.ndarray:
    return (synth_pad(seconds) + 0.9 * bass_heavy(seconds) + 0.4 * treble_bright(seconds)).astype(np.float32)


STYLES = {"bass_heavy": bass_heavy, "treble_bright": treble_bright, "synth_pad": synth_pad, "pad_and_drums": pad_and_drums}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    args = ap.parse_args()
    out = Path(args.out_dir)
    pygame.init()
    for i, (name, fn) in enumerate(STYLES.items()):
        audio = fn(12.0)
        cfg = dataclasses.replace(Config())
        stats = run(audio, SR, [(0, name.replace("_", " "))], out / name, every=3.0, cfg=cfg, start=2.0, seed=11 + i)
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in stats.items()})


if __name__ == "__main__":
    main()
