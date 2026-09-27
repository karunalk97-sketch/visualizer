"""Randomize: shuffle the look for people who don't know what they want. Every
result is a valid, watchable combination.

* `randomize(cfg)` -- across everything: which versions (one, or a fusion of two
  or three), which of their elements, and the look.
* `randomize(cfg, within=True)` -- keep the versions you picked; shuffle only their
  elements and the look.
"""
from __future__ import annotations

import numpy as np

from .config import GLYPH_SETS, PIXEL_STEPS, Config
from .engines import REGISTRY, elements_of

_SET_KEYS = [k for k, _ in GLYPH_SETS]


def _elements(rng: np.random.Generator, version: str) -> list[str]:
    keys = [k for k, _ in elements_of(version)]
    on = [k for k in keys if rng.random() < 0.7]
    return on or [str(rng.choice(keys))]


def randomize(cfg: Config, rng: np.random.Generator | None = None, within: bool = False) -> None:
    rng = rng or np.random.default_rng()
    if not within:
        keys = list(REGISTRY)
        picked = rng.choice(keys, size=int(rng.choice([1, 1, 1, 2, 2, 3])), replace=False)
        cfg.versions = [k for k in keys if k in picked]
    for v in cfg.versions:
        cfg.elements[v] = _elements(rng, v)
    chars = bool(rng.random() < 0.35)
    cfg.render_mode = "chars" if chars else "pixels"
    cfg.bit_depth = int(rng.choice([1, 1, 2, 3]))
    cfg.pixel_size = int(rng.choice([p for p in PIXEL_STEPS if p <= 6]))
    cfg.glyph_cell = int(rng.choice([8, 10, 12, 14, 16, 20]))
    picked = rng.choice(_SET_KEYS, size=int(rng.choice([1, 1, 2, 3])), replace=False)
    cfg.glyph_sets = [k for k in _SET_KEYS if k in picked]
    cfg.glyph_mapping = str(rng.choice(["random", "random", "brightness"]))
    cfg.overlap_invert = round(float(rng.choice([1.0, 1.0, rng.uniform(0.6, 1.0)])), 2)
    cfg.pixel_decay = round(float(rng.choice([0.0, 0.0, rng.uniform(0.2, 0.6)])), 2)
