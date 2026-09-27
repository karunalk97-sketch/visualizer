import numpy as np

from visualizer.spectral_field import resize_bilinear
from visualizer.waves import EDGES, WaveField, compose, swash

BINS = 96
W, H = 80, 45


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


def make_wave(edge="bottom", amp=0.8, bin_=60, rich=0.6, seed=1, **kw):
    f = WaveField(BINS, 160, 90, seed=seed, **kw)
    wave = f._spawn(amp, bin_, rich, edge=edge)
    f._waves.clear()
    return f, wave


def at(f, wave, t):
    wave.age = int(t * wave.life)
    wave.foam_off = wave.age * 0.35
    return f._draw(wave, swash(t, wave.hold))


# -- births -----------------------------------------------------------------------

def test_output_shape_and_range():
    field = WaveField(BINS, W, H, seed=1)
    out = run(field, pad(60), 300)
    assert out.shape == (H, W)
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_a_new_synth_note_brings_in_a_wave():
    field = WaveField(BINS, W, H, seed=1)
    run(field, np.zeros(BINS, dtype=np.float32), 50)
    assert field.wave_count == 0
    run(field, pad(60), 150)
    assert field.wave_count >= 1
    assert (field.update(pad(60)) > 0.02).any()


def test_waves_come_and_go_rather_than_being_constant():
    field = WaveField(BINS, W, H, seed=4)
    cov = np.array([(field.update(pad(60)) > 0.1).mean() for _ in range(2400)])
    assert (cov > 0.02).mean() < 0.9       # there are stretches with no surf
    assert (cov > 0.02).mean() > 0.2       # but it definitely comes in
    assert field.wave_count <= 2


def test_silence_fades_everything_out_including_the_wet_sand():
    field = WaveField(BINS, W, H, seed=1)
    run(field, pad(60), 250)
    out = run(field, np.zeros(BINS, dtype=np.float32), 900)
    assert out.max() < 0.02
    assert not field.active


def test_drum_hits_do_not_make_waves():
    field = WaveField(BINS, W, H, seed=2)
    silence = np.zeros(BINS, dtype=np.float32)
    for n in range(600):
        field.update(np.full(BINS, 0.6, dtype=np.float32) if n % 30 == 0 else silence)
    assert field.wave_count == 0


def test_bass_alone_makes_no_waves():
    field = WaveField(BINS, W, H, seed=3)
    run(field, pad(6, 0.8), 400)  # ~50 Hz, below the wave range
    assert field.wave_count == 0


def test_rate_setting_changes_how_often_waves_are_born():
    def births(rate):
        f = WaveField(BINS, W, H, seed=11, rate=rate)
        total = 0
        for _ in range(3000):
            before = len(f._waves)
            f.update(pad(60))
            total += max(0, len(f._waves) - before)
        return total
    assert births(2.0) > births(0.5)


def test_reshuffle_lets_waves_finish_and_rearms():
    field = WaveField(BINS, W, H, seed=1)
    run(field, pad(60), 200)
    n = field.wave_count
    field.reshuffle()
    assert field.wave_count == n
    assert field._armed.all()


# -- the surf itself --------------------------------------------------------------

def test_swash_runs_up_pauses_and_pulls_back():
    hold = 0.2
    xs = np.linspace(0, 1, 201)
    r = np.array([swash(t, hold) for t in xs])
    assert r[0] == 0.0 and r[-1] == 0.0                    # starts and ends at the shore
    assert r.min() >= 0.0 and r.max() <= 1.0
    peak = int(np.argmax(r))
    assert np.all(np.diff(r[:peak + 1]) >= -1e-9)          # up
    assert np.all(np.diff(r[peak + int(0.3 * 200):]) <= 1e-9)   # then back
    assert (r > 0.999).sum() >= 0.15 * 200                 # a real pause at the top
    up = next(t for t, v in zip(xs, r) if v > 0.9)
    down = next(t for t, v in zip(xs[::-1], r[::-1]) if v > 0.9)
    assert up < 1 - down                                    # the uprush is quicker than the backwash


def test_every_wave_spans_the_whole_edge_from_every_direction():
    for edge in EDGES:
        f, wave = make_wave(edge)
        for t in (0.03, 0.15, 0.3, 0.6, 0.85):
            sheet, foam = at(f, wave, t)
            layer = np.maximum(sheet, foam)
            axis = 0 if edge in ("bottom", "top") else 1        # collapse the depth axis: one value per position along the edge
            assert (layer.max(axis=axis) > 0.05).all(), (edge, t)


def test_the_wave_starts_as_a_line_hugging_the_edge():
    f, wave = make_wave("bottom")
    sheet, foam = at(f, wave, 0.005)
    rows = np.nonzero((foam > 0.05).any(axis=1))[0]
    assert rows.min() >= f.grid_h - 10                        # only the bottom ~10% of the screen


def test_the_shoreline_is_irregular_and_never_repeats():
    f1, w1 = make_wave("bottom", seed=1)
    f2, w2 = make_wave("bottom", seed=2)
    a, b = f1._shoreline(w1, 1.0), f2._shoreline(w2, 1.0)
    assert not np.allclose(a, b)                              # two waves differ
    assert np.std(a) > 0.01                                   # and neither is a straight line
    before = f1._shoreline(w1, 1.0).copy()
    w1.age += 60
    assert not np.allclose(before, f1._shoreline(w1, 1.0))    # the shoreline keeps shifting as the wave moves


def test_a_random_edge_each_wave_and_never_the_same_twice_in_a_row():
    f = WaveField(BINS, W, H, seed=3)
    edges = [f._spawn(0.8, 60, 0.5).edge for _ in range(60)]
    assert set(edges) == set(EDGES)
    assert all(a != b for a, b in zip(edges, edges[1:]))


def test_louder_notes_run_further_up_the_beach():
    f = WaveField(BINS, W, H, seed=5)
    quiet = np.mean([f._spawn(0.35, 60, 0.5).reach for _ in range(30)])
    loud = np.mean([f._spawn(0.95, 60, 0.5).reach for _ in range(30)])
    assert loud > quiet + 0.15


def test_low_sounds_make_wide_foam_and_bright_sounds_a_fine_line():
    f = WaveField(BINS, W, H, seed=6)
    low = np.mean([f._spawn(0.8, 34, 0.5).foam_width for _ in range(30)])
    high = np.mean([f._spawn(0.8, 90, 0.5).foam_width for _ in range(30)])
    assert low > 1.6 * high


def test_rich_harmonic_sound_makes_a_more_ragged_shoreline():
    f = WaveField(BINS, W, H, seed=7)
    plain = np.mean([f._spawn(0.8, 60, 0.05).wobble for _ in range(30)])
    rich = np.mean([f._spawn(0.8, 60, 0.95).wobble for _ in range(30)])
    assert rich > 1.5 * plain


def test_foam_is_brightest_at_the_leading_edge_and_water_thins_behind():
    f, wave = make_wave("bottom", seed=3)
    sheet, foam = at(f, wave, 0.4)
    front_rows = np.nonzero((foam > 0.3).any(axis=1))[0]
    assert len(front_rows)                                    # there is a bright foam line
    deep = foam[int(f.grid_h * 0.97):]                        # right at the shore side, well behind the front
    line = foam[front_rows.min():front_rows.min() + 6]
    assert line.mean() > 3 * deep.mean()
    top, bottom = sheet[int(0.55 * f.grid_h)], sheet[-2]
    assert sheet.max() <= 0.51 and bottom.mean() > 0           # the sheet is a partial inversion, not full


def test_the_waterline_is_soft_not_sharp():
    f, wave = make_wave("bottom", seed=4, softness=0.6)
    sheet, _ = at(f, wave, 0.4)
    assert np.abs(np.diff(sheet, axis=0)).max() < 0.2         # a gradual edge, never a hard step
    soft = WaveField(BINS, 160, 90, seed=4, softness=1.0)
    hard = WaveField(BINS, 160, 90, seed=4, softness=0.0)
    ws, wh = soft._spawn(0.8, 60, 0.5, edge="bottom"), hard._spawn(0.8, 60, 0.5, edge="bottom")
    soft._waves.clear(); hard._waves.clear()
    s_soft, _ = at(soft, ws, 0.4)
    s_hard, _ = at(hard, wh, 0.4)
    assert (np.abs(np.diff(s_soft, axis=0)).max()) < np.abs(np.diff(s_hard, axis=0)).max()


def test_the_sand_stays_faintly_wet_then_dries():
    field = WaveField(BINS, W, H, seed=1)
    run(field, pad(60), 400)
    wet_now = float(field._wet.max())
    assert wet_now > 0.05
    run(field, np.zeros(BINS, dtype=np.float32), 60)
    assert 0.0 < float(field._wet.max()) < wet_now * 1.01
    run(field, np.zeros(BINS, dtype=np.float32), 800)
    assert float(field._wet.max()) < 0.01


def test_crossing_waves_and_layers_invert():
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
