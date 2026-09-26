"""Turns raw PCM audio into a small vector of band energies (for bar-style
visualizations) using an FFT with log-spaced bands and simple frame-to-frame
decay, so bars don't jitter or snap instantly to zero.
"""
from __future__ import annotations

import numpy as np


class SpectrumAnalyzer:
    def __init__(
        self,
        sample_rate: int,
        num_bands: int = 32,
        decay: float = 0.85,
        gain: float = 1.0,
        min_freq: float = 40.0,
        max_freq: float = 16000.0,
    ) -> None:
        self.sample_rate = sample_rate
        self.num_bands = num_bands
        self.decay = decay
        self.gain = gain
        self.min_freq = min_freq
        self.max_freq = max_freq
        self._levels = np.zeros(num_bands, dtype=np.float32)
        self._band_edges: np.ndarray | None = None
        self._fft_size: int | None = None

    def _ensure_band_edges(self, fft_size: int) -> None:
        if self._band_edges is not None and self._fft_size == fft_size:
            return
        self._fft_size = fft_size
        freqs = np.fft.rfftfreq(fft_size, d=1.0 / self.sample_rate)
        lo = max(self.min_freq, freqs[1] if len(freqs) > 1 else self.min_freq)
        hi = min(self.max_freq, freqs[-1])
        # Log-spaced band edges so bass gets as much visual width as treble.
        edges_hz = np.logspace(np.log10(lo), np.log10(hi), self.num_bands + 1)
        self._band_edges = np.searchsorted(freqs, edges_hz)

    def process(self, samples: np.ndarray) -> np.ndarray:
        """samples: 1-D float32 array (mono). Returns num_bands floats in [0, 1]."""
        if samples.ndim > 1:
            samples = samples.mean(axis=1)

        window = np.hanning(len(samples))
        spectrum = np.abs(np.fft.rfft(samples * window))
        self._ensure_band_edges(len(samples))

        edges = self._band_edges
        raw = np.zeros(self.num_bands, dtype=np.float32)
        for i in range(self.num_bands):
            lo, hi = edges[i], max(edges[i + 1], edges[i] + 1)
            raw[i] = spectrum[lo:hi].mean() if hi > lo else 0.0

        # Rough loudness normalization + gain, then compress to 0..1.
        raw = raw * self.gain / (len(samples) / 2)
        raw = np.clip(np.sqrt(raw), 0.0, 1.0)  # sqrt gives a more perceptual falloff

        # Attack instantly, decay slowly (classic spectrum-analyzer behavior).
        self._levels = np.maximum(raw, self._levels * self.decay)
        return self._levels.copy()
