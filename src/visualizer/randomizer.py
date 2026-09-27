"""The Randomize button: shuffle the whole look for people who don't know what
they want. Every result is a valid, watchable combination."""
from __future__ import annotations

import numpy as np

from .config import GLYPH_SETS, PIXEL_STEPS, Config

_SET_KEYS = [k for k, _ in GLYPH_SETS]


def randomize(cfg: Config, rng: np.random.Generator | None = None) -> None:
    rng = rng or np.random.default_rng()
    chars = bool(rng.random() < 0.5)
    cfg.render_mode = "chars" if chars else "pixels"
    cfg.bit_depth = int(rng.choice([1, 1, 2, 3]))
    cfg.pixel_size = int(rng.choice([p for p in PIXEL_STEPS if p <= 6]))
    cfg.glyph_cell = int(rng.choice([8, 10, 12, 14, 16, 20]))
    picked = rng.choice(_SET_KEYS, size=int(rng.choice([1, 1, 2, 3])), replace=False)
    cfg.glyph_sets = [k for k in _SET_KEYS if k in picked]
    cfg.glyph_mapping = str(rng.choice(["random", "random", "brightness"]))
    cfg.overlap_invert = round(float(rng.uniform(0.4, 1.0)), 2)
    cfg.depth = round(float(rng.uniform(0.15, 0.75)), 2)
