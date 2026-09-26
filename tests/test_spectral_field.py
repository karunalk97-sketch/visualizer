import numpy as np

from visualizer.spectral_field import SpectralField, resize_nearest


def test_output_shape_and_range():
    field = SpectralField(num_bins=32, cluster_w=40, cluster_h=24)
    out = field.update(np.random.rand(32).astype(np.float32))
    assert out.shape == (24, 40)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_bin_positions_are_deterministic_and_spread_out():
    field = SpectralField(num_bins=64, cluster_w=50, cluster_h=30, seed=7)
    other = SpectralField(num_bins=64, cluster_w=50, cluster_h=30, seed=7)
    # same seed/config -> same frequency-to-position mapping every run
    assert np.allclose(field._bx, other._bx)
    assert np.allclose(field._by, other._by)
    # positions cover more than a single row/column (a real 2D spread, not a bar)
    assert len(set(np.round(field._bx).astype(int))) > 5
    assert len(set(np.round(field._by).astype(int))) > 5


def test_energy_decays_without_new_input():
    field = SpectralField(num_bins=16, cluster_w=30, cluster_h=20, persistence=0.5)
    loud = np.full(16, 0.9, dtype=np.float32)
    silence = np.zeros(16, dtype=np.float32)

    first = field.update(loud).sum()
    second = field.update(silence).sum()
    assert second < first


def test_only_active_bins_light_up():
    field = SpectralField(num_bins=8, cluster_w=40, cluster_h=40, persistence=0.0)
    levels = np.zeros(8, dtype=np.float32)
    levels[3] = 1.0
    out = field.update(levels)
    assert out.sum() > 0
    # most of a 40x40 field should still be dark from a single lit bin
    assert (out > 0.05).mean() < 0.2


def test_resize_nearest_shape_and_content():
    small = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    big = resize_nearest(small, out_h=4, out_w=4)
    assert big.shape == (4, 4)
    assert set(np.unique(big)) <= {0.0, 1.0}
