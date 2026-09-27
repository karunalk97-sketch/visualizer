"""The classic spectrum analyzer the early versions (V1-V3) were built on: an FFT
into 96 log-spaced bands with attack/decay smoothing."""
from __future__ import annotations

import numpy as np

NUM_BANDS = 96
MIN_FREQ, MAX_FREQ = 40.0, 16000.0


def band_centers(sample_rate: int = 48000) -> np.ndarray:
    edges = np.logspace(np.log10(MIN_FREQ), np.log10(min(MAX_FREQ, sample_rate / 2)), NUM_BANDS + 1)
    return np.sqrt(edges[:-1] * edges[1:])


class SpectrumAnalyzer:
    def __init__(self, sample_rate: int, num_bands: int = NUM_BANDS, decay: float = 0.85, gain: float = 1.0) -> None:
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.decay = decay
        self.gain = gain
        self._levels = np.zeros(num_bands, dtype=np.float32)
        self._band_edges: np.ndarray | None = None
        self._fft_size: int | None = None

    def _ensure_band_edges(self, fft_size: int) -> None:
        if self._band_edges is not None and self._fft_size == fft_size:
            return
        self._fft_size = fft_size
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        lo = max(MIN_FREQ, freqs[1] if len(freqs) > 1 else MIN_FREQ)
        hi = min(MAX_FREQ, freqs[-1])
        edges_hz = np.logspace(np.log10(lo), np.log10(hi), self.num_bands + 1)
        self._band_edges = np.searchsorted(freqs, edges_hz)

    def set_sample_rate(self, sample_rate: int) -> None:
        if sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            self._band_edges = None
            self._fft_size = None

    def process(self, samples: np.ndarray) -> np.ndarray:
        """samples: mono or (n, channels) float32. Returns num_bands floats in 0..1."""
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if len(samples) < 32:
            self._levels = self._levels * self.decay
            return self._levels.copy()
        window = np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(samples * window))
        self._ensure_band_edges(len(samples))
        edges = self._band_edges
        raw = np.zeros(self.num_bands, dtype=np.float32)
        for i in range(self.num_bands):
            lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
            raw[i] = spectrum[lo:hi].mean() if hi > lo else 0.0
        raw = raw * self.gain / (len(samples) / 2)
        raw = np.clip(np.sqrt(raw), 0.0, 1.0)
        self._levels = np.maximum(raw, self._levels * self.decay)
        return self._levels.copy()
