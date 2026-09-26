import numpy as np

from visualizer.palette import get_palette, quantize


def test_bit_depth_controls_color_count():
    assert len(get_palette("mono", 1)) == 2
    assert len(get_palette("mono", 2)) == 4
    assert len(get_palette("mono", 4)) == 16


def test_custom_palette_overrides_builtin():
    colors = get_palette("custom", 2, custom_colors=[[1, 2, 3], [4, 5, 6]])
    assert colors == [(1, 2, 3), (4, 5, 6)]


def test_quantize_output_shape_and_range():
    intensity = np.random.rand(8, 8).astype(np.float32)
    colors = get_palette("gameboy", 2)
    rgb = quantize(intensity, colors, dither=True)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    used = {tuple(px) for row in rgb for px in row}
    assert used.issubset(set(colors))


def test_quantize_extremes_map_to_first_and_last_color():
    colors = get_palette("mono", 1)
    black = quantize(np.zeros((2, 2), dtype=np.float32), colors, dither=False)
    white = quantize(np.ones((2, 2), dtype=np.float32), colors, dither=False)
    assert (black == np.array(colors[0])).all()
    assert (white == np.array(colors[-1])).all()
