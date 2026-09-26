import numpy as np

from visualizer.spectral_field import resize_bilinear
from visualizer.waves import WaveField, compose

BINS = 96


def pad(freq_bin: int, level: float = 0.5) -> np.ndarray:
    """A steady synth-like tone: energy in one bin and its neighbours."""
    lv = np.zeros(BINS, dtype=np.float32)
    lv[freq_bin - 1:freq_bin + 2] = [level * 0.6, level, level * 0.6]
    return lv


def run(field: WaveField, levels: np.ndarray, frames: int) -> np.ndarray:
    out = None
    for _ in range(frames):
        out = field.update(levels)
    return out


def test_output_shape_and_range():
    field = WaveField(BINS, 80, 45, seed=1)
    out = run(field, pad(60), 120)
    assert out.shape == (45, 80)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_sustained_tone_makes_waves():
    field = WaveField(BINS, 80, 45, seed=1)
    out = run(field, pad(60), 200)
    assert (out > 0.2).mean() > 0.03


def test_silence_is_black_and_waves_fade_out():
    field = WaveField(BINS, 80, 45, seed=1)
    run(field, pad(60), 200)
    out = run(field, np.zeros(BINS, dtype=np.float32), 400)
    assert out.max() < 0.02


def test_drum_hits_do_not_make_waves():
    """A hit that isn't sustained is not a synth: it must not conjure waves."""
    field = WaveField(BINS, 80, 45, seed=2)
    silence = np.zeros(BINS, dtype=np.float32)
    total = 0.0
    for n in range(300):
        hit = np.full(BINS, 0.6, dtype=np.float32) if n % 30 == 0 else silence
        total += float(field.update(hit).mean())
    tone = WaveField(BINS, 80, 45, seed=2)
    tone_total = sum(float(tone.update(pad(60)).mean()) for _ in range(300))
    assert total < 0.25 * tone_total


def test_bass_alone_makes_no_waves():
    field = WaveField(BINS, 80, 45, seed=3)
    out = run(field, pad(6, 0.8), 300)  # ~50 Hz, below the wave range
    assert out.max() < 0.05


def test_waves_move_over_time():
    field = WaveField(BINS, 80, 45, seed=1)
    a = run(field, pad(60), 200).copy()
    b = run(field, pad(60), 30)
    assert np.abs(a - b).mean() > 0.005


def test_different_pitch_changes_band_spacing():
    def spacing(bin_):
        f = WaveField(BINS, 160, 90, seed=5)
        out = run(f, pad(bin_), 400)
        return float(np.abs(np.diff(out > 0.3, axis=1)).mean())  # edge density ~ bands per width
    assert spacing(88) > spacing(50)   # higher notes -> tighter bands


def test_reshuffle_changes_directions():
    field = WaveField(BINS, 80, 45, seed=1)
    before = field._angle.copy()
    field.reshuffle()
    assert not np.allclose(before, field._angle)


def test_compose_inverts_instead_of_graying():
    base = np.array([[1.0, 0.0, 0.3, 0.5]], dtype=np.float32)
    full = np.ones_like(base)
    assert np.allclose(compose(base, full), [[0.0, 1.0, 0.7, 0.5]])   # white -> black, black -> white, gray flips
    assert np.allclose(compose(base, np.zeros_like(base)), base)       # no wave, no change
    half = compose(base, full * 0.5)
    assert np.allclose(half, 0.5)                                      # half strength meets in the middle


def test_bilinear_is_smooth_and_range_preserving():
    small = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    big = resize_bilinear(small, 8, 8)
    assert big.shape == (8, 8)
    assert big.min() >= 0.0 and big.max() <= 1.0
    assert len(np.unique(np.round(big, 3))) > 4   # smooth ramps, not just 0 and 1
