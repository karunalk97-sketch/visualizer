import numpy as np

from visualizer.analyzer import SpectrumAnalyzer


def test_bass_tone_lights_up_low_band():
    sr = 48000
    analyzer = SpectrumAnalyzer(sample_rate=sr, num_bands=16, decay=0.0, gain=50.0)
    t = np.arange(2048) / sr
    samples = np.sin(2 * np.pi * 100.0 * t).astype(np.float32)  # low bass tone
    levels = analyzer.process(samples)
    assert levels.shape == (16,)
    assert np.argmax(levels) < 4  # energy concentrated in the low bands


def test_treble_tone_lights_up_high_band():
    sr = 48000
    analyzer = SpectrumAnalyzer(sample_rate=sr, num_bands=16, decay=0.0, gain=50.0)
    t = np.arange(2048) / sr
    samples = np.sin(2 * np.pi * 8000.0 * t).astype(np.float32)  # high treble tone
    levels = analyzer.process(samples)
    assert np.argmax(levels) > 10  # energy concentrated in the high bands


def test_decay_falls_off_without_new_energy():
    sr = 48000
    analyzer = SpectrumAnalyzer(sample_rate=sr, num_bands=8, decay=0.5, gain=50.0)
    t = np.arange(2048) / sr
    loud = np.sin(2 * np.pi * 1000.0 * t).astype(np.float32)
    silence = np.zeros(2048, dtype=np.float32)

    first = analyzer.process(loud)
    second = analyzer.process(silence)
    assert second.sum() <= first.sum()
