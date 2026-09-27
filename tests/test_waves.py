import numpy as np

from visualizer.spectral_field import resize_bilinear
from visualizer.waves import EDGES, PROFILE_POINTS, WaveField, compose, smooth_profile

BINS = 96
W, H = 80, 45


def pad(freq_bin: int, level: float = 0.5) -> np.ndarray:
    """A steady synth-like tone: energy in one bin and its neighbours."""
    lv = np.zeros(BINS, dtype=np.float32)
    lv[freq_bin - 1:freq_bin + 2] = [level * 0.6, level, level * 0.6]
    return lv


def chord(bins=(40, 52, 63), level=0.5) -> np.ndarray:
    lv = np.zeros(BINS, dtype=np.float32)
    for b in bins:
        lv[b - 1:b + 2] = [level * 0.6, level, level * 0.6]
    return lv


def run(field: WaveField, levels: np.ndarray, frames: int) -> np.ndarray:
    out = None
    for _ in range(frames):
        out = field.update(levels)
    return out


def make_wave(edge="bottom", amp=0.8, bin_=60, rich=0.6, seed=1, sustained=None, **kw):
    f = WaveField(BINS, 160, 90, seed=seed, **kw)
    if sustained is not None:
        f._sustained = sustained.astype(np.float32)
    wave = f._spawn(amp, bin_, rich, edge=edge)
    f._waves.clear()
    return f, wave


def at(f, wave, t):
    wave.age = int(t * wave.life)
    wave.foam_off = wave.age * 0.3
    return f._draw(wave)


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
    cov = np.array([(field.update(pad(60)) > 0.1).mean() for _ in range(3000)])
    assert (cov > 0.01).mean() < 0.9       # there are stretches with no wave
    assert (cov > 0.01).mean() > 0.2       # but it definitely comes in
    assert field.wave_count <= 2


def test_silence_fades_everything_out_and_leaves_nothing_behind():
    field = WaveField(BINS, W, H, seed=1)
    run(field, pad(60), 250)
    out = run(field, np.zeros(BINS, dtype=np.float32), 200)
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
        for _ in range(4000):
            before = len(f._waves)
            f.update(pad(60))
            total += max(0, len(f._waves) - before)
        return total
    assert births(2.0) > births(0.5)


def test_reshuffle_lets_the_wave_finish_and_rearms():
    field = WaveField(BINS, W, H, seed=1)
    run(field, pad(60), 200)
    n = field.wave_count
    field.reshuffle()
    assert field.wave_count == n
    assert field._armed.all()


# -- it is one line, not a wash ---------------------------------------------------

def test_the_wave_is_a_single_thin_line_never_a_wash_of_the_screen():
    for edge in EDGES:
        f, wave = make_wave(edge, sustained=np.linspace(0.1, 0.5, BINS))
        for t in (0.15, 0.3, 0.5, 0.7, 0.85):
            layer = at(f, wave, t)
            assert (layer > 0.05).mean() < 0.35, (edge, t)        # a band, not the whole screen
            assert layer.max() > 0.3 or t in (0.15, 0.85)          # and it is clearly there mid-crossing


def test_nothing_follows_the_line_no_sheet_no_wet_sand():
    f, wave = make_wave("bottom", sustained=np.linspace(0.1, 0.5, BINS))
    layer = at(f, wave, 0.55)
    centre, thick = f._line(wave)
    dist = f._dist["bottom"]
    behind = (dist - centre) > 2.5 * thick.max()                   # the region the line has already passed
    ahead = (dist - centre) < -2.5 * thick.max()
    assert behind.any() and ahead.any()
    assert layer[behind].max() < 0.02 and layer[ahead].max() < 0.02
    assert not hasattr(f, "_wet")


def test_the_frame_is_empty_again_as_soon_as_the_wave_has_left():
    field = WaveField(BINS, W, H, seed=1)
    run(field, pad(60), 200)
    silence = np.zeros(BINS, dtype=np.float32)
    for _ in range(700):
        out = field.update(silence)
    assert field.wave_count == 0 and out.max() == 0.0


def test_the_line_spans_the_whole_edge_from_every_direction():
    for edge in EDGES:
        f, wave = make_wave(edge, sustained=np.linspace(0.1, 0.5, BINS))
        for t in (0.42, 0.5, 0.58):                                # while it is crossing the middle of the screen
            layer = at(f, wave, t)
            axis = 0 if edge in ("bottom", "top") else 1           # collapse the depth axis: one value per position along the edge
            assert (layer.max(axis=axis) > 0.05).all(), (edge, t)


def test_it_rolls_across_the_screen_once_from_one_edge_to_the_other():
    f, wave = make_wave("bottom", sustained=np.linspace(0.1, 0.5, BINS))
    pos = []
    for t in np.linspace(0.0, 1.0, 21):
        wave.age = int(t * wave.life)
        pos.append(f._position(wave))
    pos = np.array(pos)
    assert pos[0] < 0.0 and pos[-1] > 1.0                          # starts outside the source edge, ends beyond the far one
    assert np.all(np.diff(pos) > 0)                                # always moving the same way: no roll back out
    assert at(f, wave, 0.0).max() < 0.1 and at(f, wave, 0.999).max() < 0.1   # off-screen at both ends


def test_each_wave_comes_from_a_random_edge_never_the_same_twice_in_a_row():
    f = WaveField(BINS, W, H, seed=3)
    edges = [f._spawn(0.8, 60, 0.5).edge for _ in range(60)]
    assert set(edges) == set(EDGES)
    assert all(a != b for a, b in zip(edges, edges[1:]))


# -- shaped by the music ----------------------------------------------------------

def test_the_line_bulges_and_thickens_where_the_music_is_strong_and_thins_where_it_is_weak():
    sustained = np.zeros(BINS, dtype=np.float32)
    sustained[70:82] = 0.6                                          # a strong sustained band high in the range
    sustained[28:96] += 0.03
    f, wave = make_wave("bottom", sustained=sustained, seed=2)
    wave.flip = False
    wave.age = int(0.5 * wave.life)
    centre, thick = f._line(wave)
    centre, thick = centre.ravel(), thick.ravel()
    strong = int(len(thick) * ((76 - 28) / (96 - 28)))              # where bins 70-82 sit along the line
    weak = int(len(thick) * 0.08)
    assert thick[strong] > 1.8 * thick[weak]                        # thicker where the sustained energy is
    assert centre[strong] > centre[weak] + 0.05                     # and pushed further forward


def test_different_music_gives_a_different_line():
    a = np.zeros(BINS, dtype=np.float32); a[35:45] = 0.5
    b = np.zeros(BINS, dtype=np.float32); b[80:90] = 0.5
    fa, wa = make_wave("bottom", sustained=a, seed=4)
    fb, wb = make_wave("bottom", sustained=b, seed=4)
    ca, _ = fa._line(wa)
    cb, _ = fb._line(wb)
    assert not np.allclose(wa.profile, wb.profile)
    assert np.abs(ca - cb).max() > 0.05


def test_the_line_keeps_listening_as_it_travels():
    field = WaveField(BINS, W, H, seed=6)
    run(field, chord((36, 48, 60)), 200)
    wave = field._waves[0]
    before = wave.profile.copy()
    run(field, chord((70, 82, 90)), 60)
    assert field._waves and not np.allclose(before, field._waves[0].profile, atol=1e-3)


def test_louder_notes_cross_faster():
    f = WaveField(BINS, W, H, seed=5)
    quiet = np.mean([f._spawn(0.35, 60, 0.5).life for _ in range(40)])
    loud = np.mean([f._spawn(0.95, 60, 0.5).life for _ in range(40)])
    assert loud < 0.85 * quiet


def test_low_sounds_make_a_thicker_line_than_bright_sounds():
    f = WaveField(BINS, W, H, seed=6)
    low = np.mean([f._spawn(0.8, 34, 0.5).width for _ in range(30)])
    high = np.mean([f._spawn(0.8, 90, 0.5).width for _ in range(30)])
    assert low > 1.4 * high


def test_only_the_sustained_tonal_energy_shapes_it():
    """Drum-like bursts (not sustained) leave the profile flat; a held chord does not."""
    field = WaveField(BINS, W, H, seed=7)
    for n in range(120):
        field._analyse(np.full(BINS, 0.6, dtype=np.float32) if n % 30 == 0 else np.zeros(BINS, dtype=np.float32))
    assert field._sustained.max() < 0.1
    field2 = WaveField(BINS, W, H, seed=7)
    for _ in range(120):
        field2._analyse(chord())
    assert field2._sustained.max() > 0.3


# -- smoothness ---------------------------------------------------------------------

def test_the_profile_is_a_smooth_curve_not_jagged():
    rng = np.random.default_rng(3)
    spiky = rng.random(BINS).astype(np.float32)                      # about as jagged as a spectrum gets
    p = smooth_profile(spiky)
    assert p.shape == (PROFILE_POINTS,) and p.min() >= 0.0 and p.max() <= 1.0 + 1e-6
    raw = np.interp(np.linspace(0, BINS - 1, PROFILE_POINTS), np.arange(BINS), spiky)
    assert np.abs(np.diff(p, 2)).max() < 0.35 * np.abs(np.diff(raw / raw.max(), 2)).max()   # far gentler curvature


def test_the_centre_line_and_edges_are_smooth_and_soft():
    f, wave = make_wave("bottom", sustained=np.random.default_rng(1).random(BINS), seed=3)
    centre, thick = f._line(wave)
    assert np.abs(np.diff(centre.ravel(), 2)).max() < 0.02          # no kinks in the curve
    assert np.abs(np.diff(thick.ravel(), 2)).max() < 0.02
    layer = at(f, wave, 0.5)
    assert np.abs(np.diff(layer, axis=0)).max() < 0.6                # gradual across the line, never a hard step (1.0)


def test_softness_setting_feathers_the_edges():
    def steepest(softness):
        f, w = make_wave("bottom", sustained=np.linspace(0.1, 0.5, BINS), softness=softness)
        return float(np.abs(np.diff(at(f, w, 0.5), axis=0)).max())
    assert steepest(1.0) < steepest(0.0)


def test_silence_gives_a_neutral_profile_not_an_error():
    assert np.allclose(smooth_profile(np.zeros(BINS, dtype=np.float32)), 0.0)
    assert np.allclose(smooth_profile(np.full(BINS, 0.3, dtype=np.float32)), 0.5)


# -- composing ----------------------------------------------------------------------

def test_crossing_lines_and_layers_invert():
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
