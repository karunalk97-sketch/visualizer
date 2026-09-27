"""Test harness: run the app's real pipeline (features -> scene -> decay -> dither ->
status bar) over *real* music and save screenshot sequences.

    python tools/record_audio.py rec 300                    # record 5 min of system audio + track names
    python tools/render_live.py OUT --audio rec              # render it through the app's pipeline

Frames are simulated at the app's frame rate with the same latest-window audio the
app sees. Writes one PNG every --every seconds, a contact sheet per song, and prints
per-frame cost and how much of the screen was lit.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pygame  # noqa: E402

from visualizer.config import Config  # noqa: E402
from visualizer.features import FFT_SIZE, AudioFeatures  # noqa: E402
from visualizer.glyphs import GlyphField  # noqa: E402
from visualizer.main import IntensitySmoother, grid_for  # noqa: E402
from visualizer.palette import grayscale_palette, quantize_indices  # noqa: E402
from visualizer.renderer import BitmapRenderer  # noqa: E402
from visualizer.engines import Fusion  # noqa: E402

W, H = 1280, 720


def load(path: str):
    audio = np.load(path + ".npy").astype(np.float32)
    meta = json.load(open(path + ".json"))
    sr = int(meta["sample_rate"])
    timeline = [(int(e["sample"]), e["label"]) for e in meta["timeline"]]
    return audio, sr, timeline


def run(audio: np.ndarray, sr: int, timeline, out: Path, every: float, cfg: Config, start: float = 0.0,
        stop: float | None = None, seed: int = 7, tag: str = "") -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pygame.display.set_mode((W, H))
    gw, gh = grid_for(cfg, W, H)
    renderer = BitmapRenderer(gw, gh, W, H)
    feats = AudioFeatures(sr, gain=cfg.gain)
    scene = Fusion(gw, gh, sample_rate=sr, seed=seed)
    smoother = IntensitySmoother()
    glyphs = GlyphField(seed=seed)
    colors = grayscale_palette(cfg.bit_depth)
    fps = cfg.fps
    dt = 1.0 / fps
    total = len(audio) / sr
    stop = min(total, stop or total)
    labels = sorted(timeline)
    lab_i, label = 0, labels[0][1] if labels else ""

    scene.configure(cfg.versions, cfg.elements,
                    mix=dict(bass=cfg.mix_bass, lowmid=cfg.mix_lowmid, vocals=cfg.mix_vocals,
                             highmid=cfg.mix_highmid, treble=cfg.mix_treble),
                    intensity=cfg.intensity, invert=cfg.overlap_invert)
    next_shot = start + 1.0
    shots: dict[str, list] = {}
    costs, lit, flips = [], [], []
    prev_binary = None
    t = 0.0
    frame = 0
    while t < stop:
        pos = int(t * sr)
        while lab_i + 1 < len(labels) and labels[lab_i + 1][0] <= pos:
            lab_i += 1
            if labels[lab_i][1] != label:
                label = labels[lab_i][1]
                scene.reshuffle()
                feats.reset_song()
        window = audio[max(0, pos - FFT_SIZE):pos]
        c0 = time.perf_counter()
        f = feats.update(window, dt)
        if f.section_change:
            scene.new_section(f.section)
        img = scene.update(window, sr, f, dt)
        img = smoother.apply(img, cfg.pixel_decay)
        costs.append(time.perf_counter() - c0)
        if t >= start:
            binary = quantize_indices(img, len(colors)) > 0
            lit.append(float(binary.mean()))
            if prev_binary is not None:
                flips.append(float((binary != prev_binary).mean()))
            prev_binary = binary
        if t >= next_shot:
            next_shot += every
            hint = f"System audio   ·   Tab: settings   ·   {cfg.bit_depth}-bit"
            if cfg.render_mode == "chars":
                glyphs.configure(cfg.glyph_sets, cfg.glyph_cell, cfg.bit_depth)
                renderer.render_gray(glyphs.render(img, cfg.glyph_mapping), label, hint)
            else:
                renderer.render_field(img, colors, label, hint)
            name = out / f"{tag}{frame:06d}_{t:07.2f}s.png"
            pygame.image.save(pygame.display.get_surface(), str(name))
            shots.setdefault(label, []).append((t, name, dict(level=f.level, section=f.section, sustain=f.sustain,
                                                              kick=f.kick_env, hit=f.hit_env)))
        t += dt
        frame += 1
    for label, items in shots.items():
        contact_sheet(items, out / f"{tag}sheet_{_safe(label)}.png", label)
    c = np.array(costs) * 1000
    return dict(frames=len(costs), ms_mean=float(c.mean()), ms_p95=float(np.percentile(c, 95)),
                lit_mean=float(np.mean(lit)) if lit else 0.0, lit_p95=float(np.percentile(lit, 95)) if lit else 0.0,
                flip_p99=float(np.percentile(flips, 99)) if flips else 0.0, flip_max=float(np.max(flips)) if flips else 0.0,
                shots={k: len(v) for k, v in shots.items()})


def _safe(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:40] or "untitled"


def contact_sheet(items, path: Path, title: str, cols: int = 3) -> None:
    thumbs = [pygame.image.load(str(p)) for _, p, _ in items]
    tw, th = 480, 270
    rows = (len(thumbs) + cols - 1) // cols
    sheet = pygame.Surface((cols * tw, rows * (th + 18) + 30))
    sheet.fill((24, 24, 24))
    font = pygame.font.SysFont("segoeui,arial", 15)
    sheet.blit(font.render(title, True, (230, 230, 230)), (8, 6))
    for i, (thumb, (t, _, info)) in enumerate(zip(thumbs, items)):
        x, y = (i % cols) * tw, 30 + (i // cols) * (th + 18)
        sheet.blit(pygame.transform.smoothscale(thumb, (tw - 4, th - 4)), (x + 2, y + 2))
        cap = f"{t:6.1f}s  lvl {info['level']:.2f}  sec {info['section']:.2f}  sus {info['sustain']:.2f}"
        sheet.blit(font.render(cap, True, (170, 170, 170)), (x + 4, y + th))
    pygame.image.save(sheet, str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--audio", help="recording path without extension (.npy + .json)")
    ap.add_argument("--every", type=float, default=2.0)
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--stop", type=float, default=None)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--set", action="append", default=[], help="config override, e.g. --set gain=0.8")
    ap.add_argument("--tag", default="")
    ap.add_argument("--versions", default="v7", help="one or more versions to fuse, e.g. v1,v4")
    args = ap.parse_args()
    pygame.init()
    cfg = Config()
    cfg.versions = args.versions.split(",")
    for kv in args.set:
        k, v = kv.split("=", 1)
        cur = getattr(cfg, k)
        cfg = dataclasses.replace(cfg, **{k: type(cur)(v) if not isinstance(cur, list) else v.split(",")})
    audio, sr, timeline = load(args.audio)
    stats = run(audio, sr, timeline, Path(args.out), args.every, cfg, args.start, args.stop, args.seed, tag=args.tag)
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
