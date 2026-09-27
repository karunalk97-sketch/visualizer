"""Run any number of versions at once and fuse them.

Every version sees the same audio, shaped the same way by the Mix (one slider per
frequency range) and by Intensity (subtle .. jarring). Their pictures are combined
the way everything else combines: where they overlap they invert. A version that is
switched off costs nothing; one that is switched back on picks up where it left off.
"""
from __future__ import annotations

import dataclasses

import numpy as np

from ..features import RANGES, Features, band_edges
from .analyzer import SpectrumAnalyzer, band_centers
from .base import Engine, Inputs
from .v1_spots import SpotsEngine
from .v2_shapes import ShapesEngine
from .v3_adaptive import AdaptiveEngine
from .v4_weather import WeatherEngine
from .v5_bursts import BurstsEngine
from .v6_waves import InkWavesEngine
from .v7_field import FieldEngine

REGISTRY: dict[str, type[Engine]] = {cls.KEY: cls for cls in (
    SpotsEngine, ShapesEngine, AdaptiveEngine, WeatherEngine, BurstsEngine, InkWavesEngine, FieldEngine)}
VERSIONS: list[tuple[str, str]] = [(k, cls.LABEL) for k, cls in REGISTRY.items()]
MIX_KEYS = tuple(r[0] for r in RANGES)                  # bass, lowmid, vocals, highmid, treble
_CLASSIC = {"v1", "v2", "v3"}                            # versions built on the classic 96-band analyzer


def elements_of(version: str) -> list[tuple[str, str]]:
    return list(REGISTRY[version].ELEMENTS)


def default_elements() -> dict[str, list[str]]:
    return {k: [e for e, _ in cls.ELEMENTS] for k, cls in REGISTRY.items()}


def _range_of_hz(hz: np.ndarray) -> np.ndarray:
    bounds = np.array([r[2] for r in RANGES[:-1]])       # 150, 500, 2000, 6000
    return np.searchsorted(bounds, hz)


def scale_features(f: Features, mix: dict, intensity: float) -> Features:
    """The same music, turned up or down per range (Mix), and made more subtle or more
    jarring (Intensity: 0 = only slow, sustained forms; 1 = normal; 2 = hits hit twice as hard)."""
    m = [float(mix.get(k, 1.0)) for k in MIX_KEYS]
    _, rng_of = band_edges()
    per_band = np.array([m[i] for i in rng_of], dtype=np.float32)
    events = intensity
    size = 0.5 + 0.5 * intensity
    snare = 0.5 * (m[3] + m[4])
    clip = lambda x: float(min(1.5, max(0.0, x)))  # noqa: E731
    return dataclasses.replace(
        f,
        bands=np.clip(np.asarray(f.bands, dtype=np.float32) * per_band * size, 0.0, 1.5),
        kick=clip(f.kick * m[0] * events), kick_env=clip(f.kick_env * m[0] * events),
        hit=clip(f.hit * snare * events), hit_env=clip(f.hit_env * snare * events),
        low=clip(f.low * m[0]), mid=clip(f.mid * 0.5 * (m[1] + m[2])), high=clip(f.high * m[4]),
        sustain=clip(f.sustain * m[2] * (0.8 + 0.2 * intensity)),
        level=clip(f.level * (0.7 + 0.3 * intensity)),
    )


class Fusion:
    def __init__(self, width: int, height: int, sample_rate: int = 48000, seed: int | None = None) -> None:
        self.W, self.H = width, height
        self._seed = np.random.default_rng(seed)
        self.analyzer = SpectrumAnalyzer(sample_rate)
        self._bin_range = _range_of_hz(band_centers(sample_rate))
        self.engines: dict[str, Engine] = {}
        self.versions: list[str] = ["v7"]
        self.elements: dict[str, list[str]] = default_elements()
        self.mix = {k: 1.0 for k in MIX_KEYS}
        self.intensity = 1.0
        self.invert = 1.0

    def configure(self, versions, elements=None, mix=None, intensity=None, invert=None) -> None:
        self.versions = [v for v in versions if v in REGISTRY] or ["v7"]
        if elements is not None:
            self.elements = {k: list(elements.get(k, self.elements.get(k, []))) for k in REGISTRY}
        if mix is not None:
            self.mix.update(mix)
        if intensity is not None:
            self.intensity = float(intensity)
        if invert is not None:
            self.invert = float(invert)

    def engine(self, key: str) -> Engine:
        if key not in self.engines:
            self.engines[key] = REGISTRY[key](self.W, self.H, seed=int(self._seed.integers(0, 2**31 - 1)))
        return self.engines[key]

    def resize(self, width: int, height: int) -> None:
        self.W, self.H = width, height
        for e in self.engines.values():
            e.resize(width, height)

    def reshuffle(self) -> None:
        for e in self.engines.values():
            e.reshuffle()

    def new_section(self, section: float) -> None:
        for e in self.engines.values():
            e.new_section(section)

    def update(self, samples: np.ndarray, sample_rate: int, f: Features, dt: float) -> np.ndarray:
        if any(v in _CLASSIC for v in self.versions):
            if sample_rate != self.analyzer.sample_rate:
                self.analyzer.set_sample_rate(sample_rate)
                self._bin_range = _range_of_hz(band_centers(sample_rate))
            bands96 = self.analyzer.process(samples)
        else:
            bands96 = np.zeros(96, dtype=np.float32)
        m = np.array([self.mix.get(k, 1.0) for k in MIX_KEYS], dtype=np.float32)
        bin_gain = m[self._bin_range] * (0.35 + 0.65 * self.intensity)
        inp = Inputs(features=scale_features(f, self.mix, self.intensity), band_levels=bands96,
                     bin_gain=bin_gain.astype(np.float32), intensity=self.intensity)
        img = None
        for key in self.versions:
            e = self.engine(key)
            e.set_elements(self.elements.get(key, []))
            e.invert = self.invert
            out = e.update(inp, dt)
            img = out if img is None else img + out - (2.0 * self.invert) * img * out
        img = np.zeros((self.H, self.W), dtype=np.float32) if img is None else img
        np.clip(img, 0.0, 1.0, out=img)
        return img
