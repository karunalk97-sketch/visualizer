import numpy as np

from visualizer.spectral_field import SpectralField, resize_nearest


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
    field = SpectralField(num_bins=64, cluster_w=100, cluster_h=60, spots_per_bin=1, seed=3)
    xy = field.positions()[np.argsort(field._owner)]  # spot position per bin, in frequency order
    step = np.linalg.norm(np.diff(xy, axis=0), axis=1).mean()
    rand_pairs = np.linalg.norm(xy[np.random.default_rng(0).permutation(64)] - xy, axis=1).mean()
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
