import numpy as np

from visualizer.fields import build_field


def test_field_shape_and_range():
    band_levels = np.random.rand(32).astype(np.float32)
    field = build_field(band_levels, grid_w=64, grid_h=32, t=1.23)
    assert field.shape == (32, 64)
    assert field.min() >= 0.0
    assert field.max() <= 1.0


def test_louder_bands_increase_average_intensity():
    quiet = np.full(32, 0.05, dtype=np.float32)
    loud = np.full(32, 0.9, dtype=np.float32)
    quiet_field = build_field(quiet, grid_w=64, grid_h=32, t=0.0)
    loud_field = build_field(loud, grid_w=64, grid_h=32, t=0.0)
    assert loud_field.mean() > quiet_field.mean()


def test_taller_bars_reach_higher_rows_in_louder_columns():
    band_levels = np.array([0.1, 0.9], dtype=np.float32)
    field = build_field(band_levels, grid_w=2, grid_h=20, t=0.0)
    # column 1 (loud) should be lit (above the ~0.85 bar-mask floor) higher up
    # the screen (lower row index) than column 0 (quiet).
    lit_rows_col0 = np.where(field[:, 0] > 0.8)[0]
    lit_rows_col1 = np.where(field[:, 1] > 0.8)[0]
    assert lit_rows_col1.min() < lit_rows_col0.min()
