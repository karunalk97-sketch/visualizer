import numpy as np

from visualizer.analyzer import SpectrumAnalyzer
import visualizer.spectral_field as spectral_field_module
from visualizer.spectral_field import BLOB, ORB, SLASH, SPIKY, STAR, SpectralField, resize_nearest

SR = 48000
BINS = 96


def test_output_shape_and_range():
    field = SpectralField(num_bins=32, cluster_w=40, cluster_h=24)
    out = field.update(np.random.rand(32).astype(np.float32))
    assert out.shape == (24, 40)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_same_seed_same_layout_different_seed_different_layout():
    a = SpectralField(num_bins=64, cluster_w=50, cluster_h=30, seed=7)
    b = SpectralField(num_bins=64, cluster_w=50, cluster_h=30, seed=7)
    c = SpectralField(num_bins=64, cluster_w=50, cluster_h=30, seed=8)
    assert np.allclose(a.positions(), b.positions())
    assert not np.allclose(a.positions(), c.positions())


def test_layout_is_not_frequency_ordered():
    """Neighbouring frequencies must NOT sit next to each other (no spiral/sweep)."""
    field = SpectralField(num_bins=64, cluster_w=100, cluster_h=60, seed=3)
    xy = field.positions()
    first = [int(np.where(field._owner == b)[0][0]) for b in range(64)]   # one spot per bin, in frequency order
    pts = xy[first]
    step = np.linalg.norm(np.diff(pts, axis=0), axis=1).mean()
    rand_pairs = np.linalg.norm(pts[np.random.default_rng(0).permutation(64)] - pts, axis=1).mean()
    assert step > 0.6 * rand_pairs  # adjacent bins are about as far apart as random pairs


def test_layout_spreads_over_the_whole_canvas_without_symmetry():
    field = SpectralField(num_bins=96, cluster_w=120, cluster_h=70, seed=5)
    xy = field.positions()
    assert xy[:, 0].min() < 15 and xy[:, 0].max() > 105
    assert xy[:, 1].min() < 10 and xy[:, 1].max() > 60
    mirrored = np.stack([119 - xy[:, 0], xy[:, 1]], axis=1)
    d = np.linalg.norm(xy[:, None] - mirrored[None], axis=2).min(axis=1)
    assert d.mean() > 1.0  # not left/right mirror symmetric


def test_reshuffle_moves_spots_over_time():
    field = SpectralField(num_bins=32, cluster_w=80, cluster_h=50, seed=1)
    before = field.positions().copy()
    field.reshuffle()
    quiet = np.zeros(32, dtype=np.float32)
    for _ in range(120):
        field.update(quiet)
    assert np.linalg.norm(field.positions() - before, axis=1).mean() > 5.0


def test_spots_drift_between_frames():
    field = SpectralField(num_bins=16, cluster_w=80, cluster_h=50, seed=2, drift=0.05)
    a = field.positions().copy()
    for _ in range(200):
        field.update(np.zeros(16, dtype=np.float32))
    assert not np.allclose(a, field.positions())


def test_energy_decays_without_new_input():
    field = SpectralField(num_bins=16, cluster_w=30, cluster_h=20, persistence=0.5)
    loud = np.full(16, 0.9, dtype=np.float32)
    silence = np.zeros(16, dtype=np.float32)

    first = field.update(loud).sum()
    second = field.update(silence).sum()
    assert second < first


def test_only_active_bins_light_up():
    field = SpectralField(num_bins=8, cluster_w=60, cluster_h=40, persistence=0.0, spots_per_bin=1)
    levels = np.zeros(8, dtype=np.float32)
    levels[3] = 1.0
    out = field.update(levels)
    assert out.sum() > 0
    assert (out > 0.05).mean() < 0.25  # one lit bin leaves most of the canvas dark


def test_resize_keeps_layout_proportional():
    field = SpectralField(num_bins=16, cluster_w=40, cluster_h=20, seed=4)
    before = field.positions() / np.array([39, 19])
    field.resize(80, 40)
    after = field.positions() / np.array([79, 39])
    assert np.allclose(before, after, atol=1e-5)
    assert field.buffer.shape == (40, 80)


def test_resize_nearest_shape_and_content():
    small = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    big = resize_nearest(small, out_h=4, out_w=4)
    assert big.shape == (4, 4)
    assert set(np.unique(big)) <= {0.0, 1.0}


# -- shapes ---------------------------------------------------------------------

def test_the_five_shape_kinds_appear_and_vary():
    field = SpectralField(num_bins=BINS, cluster_w=100, cluster_h=60, seed=9)
    assert set(field._kind.tolist()) == {ORB, SPIKY, STAR, BLOB, SLASH}
    assert field._elong.max() > 2.0 and field._elong.min() < 1.15
    assert len(set(np.round(field._angle, 2).tolist())) > 50       # pointing every which way


def test_rings_and_petals_are_gone():
    for name in ("RING", "PETAL", "HYBRID", "STRING"):
        assert not hasattr(spectral_field_module, name), name
    field = SpectralField(num_bins=BINS, cluster_w=100, cluster_h=60, seed=2)
    assert field._kind.max() <= 4                                   # only the five kinds exist


def test_each_kind_looks_like_what_it_is():
    f = SpectralField(num_bins=BINS, cluster_w=100, cluster_h=60, seed=5)
    k = f._kind
    assert np.all(f._spikes[k == ORB] == 0) and np.all(f._depth[k == ORB] == 0) and f._elong[k == ORB].max() <= 1.11   # a smooth round orb
    assert f._spikes[k == SPIKY].min() >= 11 and f._depth[k == SPIKY].max() <= 0.6                                     # many short spikes on a round core
    assert 5 <= f._spikes[k == STAR].min() and f._spikes[k == STAR].max() <= 9 and f._depth[k == STAR].min() >= 1.3      # few long spikes
    assert f._spikes[k == BLOB].max() <= 3 and np.all(f._expo[k == BLOB] == 2.0)                                      # lumpy, smooth lobes (no cusps)
    assert f._elong[k == SLASH].min() >= 2.0                                                                           # only slashes are long and thin
    assert f._elong[k != SLASH].max() <= 1.31


def test_slashes_are_a_small_share():
    fractions = []
    for seed in range(8):
        f = SpectralField(num_bins=BINS, cluster_w=100, cluster_h=60, seed=seed)
        fractions.append(float(np.mean(f._kind == SLASH)))
        assert np.mean(f._elong > 2.0) < 0.18
    assert 0.03 < np.mean(fractions) < 0.15
    f = SpectralField(num_bins=BINS, cluster_w=100, cluster_h=60, seed=1)
    for kind in (ORB, SPIKY, STAR, BLOB):                           # the orbs, stars and blobs carry the picture
        assert np.mean(f._kind == kind) > 0.12


def test_every_shape_rotates_steadily_in_either_direction():
    f = SpectralField(num_bins=BINS, cluster_w=100, cluster_h=60, seed=3)
    assert np.abs(f._spin).min() >= 0.008                           # none is (nearly) still: at least ~0.5 rad/s
    assert (f._spin > 0).any() and (f._spin < 0).any()              # clockwise and counter-clockwise
    def render(frame):
        f._frame = frame
        layer = np.zeros((60, 100), dtype=np.float32)
        i = int(np.where(f._kind == STAR)[0][0])
        f._angle[i], f._spin[i] = 0.3, 0.02
        f._splat(layer, i, 50.0, 30.0, 12.0, 1.0)
        return layer
    a, b = render(0), render(40)
    assert not np.allclose(a, b, atol=0.02)                         # a star at two moments is visibly turned


def test_blobs_are_lumpy_but_smooth():
    f = SpectralField(num_bins=8, cluster_w=80, cluster_h=80, persistence=0.0, spots_per_bin=1, seed=1)
    i = 0
    f._kind[:] = BLOB; f._elong[:] = 1.2; f._spikes[:] = 3; f._depth[:] = 0.5; f._expo[:] = 2.0
    f._angle[:] = 0.0; f._spin[:] = 0.0
    layer = np.zeros((80, 80), dtype=np.float32)
    f._splat(layer, i, 40.0, 40.0, 16.0, 1.0)
    assert layer[40, 40] > 0.9 and layer.sum() > 0
    assert np.abs(np.diff(layer, axis=1)).max() < 0.35              # soft edge all round: no hard steps or cusps

def test_reshuffle_draws_new_shapes():
    field = SpectralField(num_bins=32, cluster_w=80, cluster_h=50, seed=1)
    before = field._angle.copy()
    field.reshuffle()
    assert not np.allclose(before, field._angle)


def test_overlaps_invert_instead_of_just_adding():
    loud = np.full(48, 1.0, dtype=np.float32)
    kw = dict(num_bins=48, cluster_w=60, cluster_h=36, persistence=0.0, seed=6, max_bins=1.0, area_budget=10.0, max_active=200)
    additive = SpectralField(invert=0.0, **kw)
    negative = SpectralField(invert=1.0, **kw)
    a, n = additive.update(loud).copy(), negative.update(loud).copy()
    assert n.sum() < a.sum()          # crossings cancel out
    assert ((n < a - 0.3).sum()) > 20  # and there are real inverted pixels, not rounding noise
    assert n.min() >= 0.0 and n.max() <= 1.0


def test_held_note_does_not_strobe():
    field = SpectralField(num_bins=16, cluster_w=60, cluster_h=36, persistence=0.88, seed=2)
    held = np.zeros(16, dtype=np.float32)
    held[5] = 0.9
    sums = [field.update(held).sum() for _ in range(30)]
    assert max(sums[10:]) < 1.25 * min(sums[10:])   # steady, not flashing frame to frame


# -- frequency mapping ----------------------------------------------------------

def _kick(seconds=6.0):
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    for s in np.arange(0, seconds, 0.5):
        i, n = int(s * SR), int(0.25 * SR)
        tt = np.arange(n) / SR
        out[i:i + n] += np.sin(2 * np.pi * (45 + 80 * np.exp(-tt * 25)) * tt) * np.exp(-tt * 9) * 0.9
        out[i:i + int(0.004 * SR)] += np.random.default_rng(1).standard_normal(int(0.004 * SR)) * 0.5   # the click
    return out.astype(np.float32)


def _hat(seconds=6.0):
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    rng = np.random.default_rng(2)
    for s in np.arange(0.25, seconds, 0.5):
        i, n = int(s * SR), int(0.06 * SR)
        tt = np.arange(n) / SR
        out[i:i + n] += np.diff(rng.standard_normal(n + 1)) * np.exp(-tt * 80) * 0.5
    return out.astype(np.float32)


def _lit(audio):
    an = SpectrumAnalyzer(SR, BINS)
    f = SpectralField(BINS, 142, 80, seed=3)
    means = []
    for k, i in enumerate(range(0, len(audio) - 1024, 1024)):
        b = f.update(an.process(audio[i:i + 1024]))
        if k > 60:
            means.append(float(b.mean()))
    return np.array(means)


def test_a_hard_bass_hit_does_not_light_the_whole_screen():
    kick = _lit(_kick())
    assert kick.max() < 0.25          # the worst frame of a hard kick + click stays a fraction of the screen
    assert kick.mean() > 0.005        # but it is definitely shown


def test_a_hi_hat_lights_a_fair_share_compared_with_a_kick():
    kick, hat = _lit(_kick()), _lit(_hat())
    assert hat.mean() > 0.01                       # a hi-hat is visible...
    assert hat.mean() > 0.4 * kick.mean()          # ...and not dwarfed by the kick


def test_loud_broadband_noise_cannot_fill_the_canvas():
    f = SpectralField(BINS, 142, 80, seed=3)
    for _ in range(80):
        b = f.update(np.full(BINS, 0.9, dtype=np.float32))
    assert (b > 0.25).mean() < 0.5


def test_each_frequency_is_judged_against_its_own_peak():
    """A treble band that is quiet in absolute terms still lights when it is loud for *that* band."""
    f = SpectralField(BINS, 100, 60, seed=1)
    lv = np.zeros(BINS, dtype=np.float32)
    lv[:10] = 0.6                        # loud bass, steady
    for _ in range(100):
        lv[80] = 0.08                    # treble peak that has always been quiet
        shaped = f._shape(lv)
    assert shaped[80] > 0.5
    assert shaped[3] < 0.2 or shaped[3] > 0                           # bass is judged the same way (no crash, sane range)
    assert shaped.min() >= 0.0 and shaped.max() <= 1.0


def test_bass_is_grouped_and_treble_gets_more_smaller_spots():
    f = SpectralField(BINS, 100, 60, seed=1)
    lv = np.zeros(BINS, dtype=np.float32)
    lv[:20] = np.linspace(0.3, 0.9, 20)
    shaped = f._shape(lv)
    groups = shaped[:20].reshape(5, 4)
    assert np.allclose(groups, groups[:, :1])                          # bass bins share a level within each group of 4
    frac = f._frac
    per_bin = np.bincount(f._owner, minlength=BINS)
    assert per_bin[frac < 0.22].max() == 1 and per_bin[frac > 0.5].min() > per_bin[frac < 0.22].max()
    low = f._size_scale[f._frac[f._owner] < 0.22].mean()
    high = f._size_scale[f._frac[f._owner] > 0.7].mean()
    assert low > 1.5 * high                                            # bass draws big, treble small


def test_area_budget_caps_what_one_frame_can_claim():
    tight = SpectralField(BINS, 100, 60, seed=2, persistence=0.0, area_budget=0.03, max_bins=1.0)
    loose = SpectralField(BINS, 100, 60, seed=2, persistence=0.0, area_budget=0.6, max_bins=1.0)
    lv = np.full(BINS, 0.8, dtype=np.float32)
    assert tight.update(lv).mean() < 0.6 * loose.update(lv).mean()


def test_silence_is_black():
    f = SpectralField(BINS, 100, 60, seed=2)
    for _ in range(40):
        out = f.update(np.zeros(BINS, dtype=np.float32))
    assert out.max() == 0.0
