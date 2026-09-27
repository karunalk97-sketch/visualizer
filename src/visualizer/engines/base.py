"""What every version of the picture has in common.

A version ("engine") turns this frame's audio into an (H, W) image in 0..1 at the
cell grid's resolution. It is made of named *elements* (terrain, ink, stars...)
that can each be switched on or off. The fusion layer runs any number of versions
together and inverts them where they overlap.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..features import Features


@dataclass
class Inputs:
    """One frame's audio, as every version may want it."""
    features: Features          # absolute features, already scaled by the mix and Intensity
    band_levels: np.ndarray     # the classic 96-band spectrum (the early versions' input)
    bin_gain: np.ndarray        # mix x Intensity for each of those 96 bands
    intensity: float            # 0 (subtle) .. 2 (jarring), 1 = normal


class Engine:
    KEY = ""
    LABEL = ""
    ELEMENTS: tuple[tuple[str, str], ...] = ()

    def __init__(self, width: int, height: int, seed: int | None = None) -> None:
        self.enabled: set[str] = {k for k, _ in self.ELEMENTS}
        self.invert = 1.0
        self.rng = np.random.default_rng(seed)
        self.W, self.H = max(8, int(width)), max(6, int(height))

    def on(self, element: str) -> bool:
        return element in self.enabled

    def set_elements(self, keys) -> None:
        self.enabled = {k for k in keys if k in dict(self.ELEMENTS)}

    def resize(self, width: int, height: int) -> None:
        self.W, self.H = max(8, int(width)), max(6, int(height))

    def reshuffle(self) -> None:
        """A new song."""

    def new_section(self, section: float) -> None:
        """The song moved into a new part."""

    def update(self, inp: Inputs, dt: float) -> np.ndarray:
        raise NotImplementedError

    def blank(self) -> np.ndarray:
        return np.zeros((self.H, self.W), dtype=np.float32)

    def xor(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Where both are lit they invert each other instead of adding up."""
        return a + b - (2.0 * self.invert) * a * b
