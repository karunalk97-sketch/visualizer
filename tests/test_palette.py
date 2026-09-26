import numpy as np

from visualizer.palette import grayscale_palette, quantize


def test_bit_depth_controls_gray_level_count():
    assert len(grayscale_palette(1)) == 2
    assert len(grayscale_palette(2)) == 4
    assert len(grayscale_palette(4)) == 16


def test_palette_is_always_grayscale():
    for bit_depth in (1, 2, 3, 4):
        for r, g, b in grayscale_palette(bit_depth):
            assert r == g == b


def test_quantize_output_shape_and_range():
    intensity = np.random.rand(8, 8).astype(np.float32)
    colors = grayscale_palette(2)
    rgb = quantize(intensity, colors)
    assert rgb.shape == (8, 8, 3)
    assert rgb.dtype == np.uint8
    used = {tuple(px) for row in rgb for px in row}
    assert used.issubset(set(colors))


def test_quantize_zero_intensity_is_solid_darkest():
    # 0 perturbed by dithering can only ever round back down to the darkest
    # color (the math can't push it up a full level), so this must be exact.
    colors = grayscale_palette(2)
    black = quantize(np.zeros((8, 8), dtype=np.float32), colors)
    assert (black == np.array(colors[0])).all()


def test_quantize_max_intensity_is_mostly_brightest():
    # ordered dithering can round one specific dither-matrix cell down a
    # level even at full intensity (that's what makes it dithering); assert
    # the overwhelming majority still hit the brightest color.
    colors = grayscale_palette(2)
    white = quantize(np.ones((8, 8), dtype=np.float32), colors)
    match = np.all(white == np.array(colors[-1]), axis=-1)
    assert match.mean() >= 0.9
