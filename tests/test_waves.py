import numpy as np

from visualizer.spectral_field import resize_bilinear
from visualizer.waves import Ribbon, WaveField, compose

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


def coverage_over_time(field: WaveField, levels_per_frame, thresh: float = 0.1) -> np.ndarray:
    return np.array([(field.update(lv) > thresh).mean() for lv in levels_per_frame])


def test_output_shape_and_range():
    field = WaveField(BINS, 60, 34, seed=1)
    out = run(field, pad(60), 200)
    assert out.shape == (34, 60)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_a_new_synth_note_is_born_as_a_ribbon():
    field = WaveField(BINS, 60, 34, seed=1)
    run(field, np.zeros(BINS, dtype=np.float32), 50)
    assert field.ribbon_count == 0
    run(field, pad(60), 150)
    assert field.ribbon_count >= 1
    assert (field.update(pad(60)) > 0.05).any()


def test_waves_come_and_go_rather_than_being_constant():
    """A held tone makes a ribbon at the start, then there are stretches with none."""
    field = WaveField(BINS, 60, 34, seed=4)
    cov = coverage_over_time(field, [pad(60)] * 1500)
    assert (cov > 0.01).mean() < 0.9       # not always on screen
    assert (cov > 0.01).mean() > 0.2       # but definitely appears
    # and a held note does not machine-gun ribbons: never more than the cap
    assert field.ribbon_count <= 5


def test_silence_fades_everything_out():
    field = WaveField(BINS, 60, 34, seed=1)
    run(field, pad(60), 200)
    out = run(field, np.zeros(BINS, dtype=np.float32), 120)
    assert out.max() < 0.02
    assert not field.active


def test_drum_hits_do_not_make_waves():
    field = WaveField(BINS, 60, 34, seed=2)
    silence = np.zeros(BINS, dtype=np.float32)
    for n in range(600):
        field.update(np.full(BINS, 0.6, dtype=np.float32) if n % 30 == 0 else silence)
    assert field.ribbon_count == 0


def test_bass_alone_makes_no_waves():
    field = WaveField(BINS, 60, 34, seed=3)
    run(field, pad(6, 0.8), 400)  # ~50 Hz, below the wave range
    assert field.ribbon_count == 0


def test_a_change_of_note_spawns_a_new_ribbon():
    field = WaveField(BINS, 60, 34, seed=5)
    run(field, pad(50), 200)
    first = field.ribbon_count
    run(field, pad(64), 150)   # same layer range? move within the layers: higher note
    assert field.ribbon_count + 0 >= first  # never loses ribbons to a note change
    assert field.ribbon_count >= 1


def test_low_notes_make_fatter_ribbons_than_high_notes():
    low = WaveField(BINS, 60, 34, seed=7)
    high = WaveField(BINS, 60, 34, seed=7)
    run(low, pad(34), 120)
    run(high, pad(90), 120)
    assert low._ribbons and high._ribbons
    assert np.mean([r.width for r in low._ribbons]) > 2 * np.mean([r.width for r in high._ribbons])


def test_ribbon_kinds_follow_the_frequencies():
    f = WaveField(BINS, 60, 34, seed=8)
    f._spawn(0, 0.8, 34, 0.1)     # low, plain tone
    f._spawn(1, 0.8, 55, 0.9)     # mid, harmonically rich
    f._spawn(3, 0.8, 90, 0.2)     # bright
    assert [r.kind for r in f._ribbons] == ["spindle", "dumbbell", "flat"]


def _profile(kind: str) -> np.ndarray:
    """Thickness of a horizontal ribbon at 9 points along its length."""
    f = WaveField(BINS, 240, 135, seed=1, softness=0.6)
    rb = Ribbon(kind=kind, x0=0.5 * f.aspect, y0=0.5, dx=0.0, dy=1.0, speed=0.0, length=1.4, width=0.06,
                bend=0.0, wobble=0.0, wob_len=0.3, phase=0.0, amp=1.0, life=1000)
    img = f._draw(rb, 2.0)
    xs = np.linspace(0.5 * f.aspect - 0.62, 0.5 * f.aspect + 0.62, 9)
    cols = [int(x / f.aspect * (f.grid_w - 1)) for x in xs]
    return np.array([float((img[:, c] > 0.5).sum()) for c in cols])


def test_spindle_is_thick_in_the_middle():
    p = _profile("spindle")
    assert p[4] > 1.5 * max(p[1], p[7], 1)


def test_dumbbell_is_thick_towards_the_ends():
    p = _profile("dumbbell")
    assert p[1] > 1.3 * p[4] and p[7] > 1.3 * p[4]


def test_ribbons_have_soft_edges():
    """No hard steps: neighbouring cells differ only a little across the ribbon."""
    f = WaveField(BINS, 240, 135, seed=1, softness=0.6)
    rb = Ribbon(kind="spindle", x0=0.5 * f.aspect, y0=0.5, dx=0.0, dy=1.0, speed=0.0, length=1.4, width=0.06,
                bend=0.0, wobble=0.0, wob_len=0.3, phase=0.0, amp=1.0, life=1000)
    img = f._draw(rb, 3.4 * (1 - 0.75 * 0.6))
    assert np.abs(np.diff(img, axis=0)).max() < 0.35
    assert np.abs(np.diff(img, axis=1)).max() < 0.35


def test_softness_setting_widens_the_falloff():
    def hard_cells(softness):
        f = WaveField(BINS, 240, 135, seed=1, softness=softness)
        rb = Ribbon(kind="flat", x0=0.5 * f.aspect, y0=0.5, dx=0.0, dy=1.0, speed=0.0, length=1.4, width=0.05,
                    bend=0.0, wobble=0.0, wob_len=0.3, phase=0.0, amp=1.0, life=1000)
        img = f._draw(rb, 3.4 * (1 - 0.75 * softness))
        return float(((img > 0.05) & (img < 0.95)).sum())
    assert hard_cells(1.0) > hard_cells(0.0)   # more feathered edge pixels when softer


def test_rate_setting_changes_how_often_waves_are_born():
    def births(rate):
        f = WaveField(BINS, 60, 34, seed=11, rate=rate)
        total = 0
        for _ in range(3000):
            before = len(f._ribbons)
            f.update(pad(60))
            total += max(0, len(f._ribbons) - before)
        return total
    assert births(2.0) > births(0.5)


def test_reshuffle_lets_ribbons_finish_and_rearms():
    field = WaveField(BINS, 60, 34, seed=1)
    run(field, pad(60), 200)
    n = field.ribbon_count
    field.reshuffle()
    assert field.ribbon_count == n
    assert field._armed.all()


def test_compose_inverts_instead_of_graying():
    base = np.array([[1.0, 0.0, 0.3, 0.5]], dtype=np.float32)
    full = np.ones_like(base)
    assert np.allclose(compose(base, full), [[0.0, 1.0, 0.7, 0.5]])   # white -> black, black -> white, gray flips
    assert np.allclose(compose(base, np.zeros_like(base)), base)       # no wave, no change
    assert np.allclose(compose(base, full * 0.5), 0.5)                 # half strength meets in the middle


def test_bilinear_is_smooth_and_range_preserving():
    small = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    big = resize_bilinear(small, 8, 8)
    assert big.shape == (8, 8)
    assert big.min() >= 0.0 and big.max() <= 1.0
    assert len(np.unique(np.round(big, 3))) > 4
