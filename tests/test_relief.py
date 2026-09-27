import numpy as np

from visualizer.relief import relief
from visualizer.spectral_field import resize_bilinear


def bump(h=60, w=100):
    """A soft hill whose size scales with the picture, like real shapes do."""
    y, x = np.mgrid[0:h, 0:w]
    return np.exp(-(((x - w / 2) / (0.14 * w)) ** 2 + ((y - h / 2) / (0.2 * h)) ** 2)).astype(np.float32)


def test_zero_amount_changes_nothing():
    b = bump()
    assert np.array_equal(relief(b, 0.0), b)


def test_output_stays_in_range_and_shape():
    out = relief(bump(), 1.0)
    assert out.shape == (60, 100) and out.min() >= 0.0 and out.max() <= 1.0


def test_a_flat_picture_stays_flat():
    flat = np.full((40, 60), 0.4, np.float32)
    assert np.allclose(relief(flat, 1.0), flat, atol=1e-4)


def test_it_lights_the_bump_from_the_upper_left():
    b = bump()
    out = relief(b, 0.8)
    shade = out - b
    h, w = b.shape
    up_left = shade[h // 2 - 12:h // 2 - 2, w // 2 - 18:w // 2 - 4].mean()
    low_right = shade[h // 2 + 2:h // 2 + 12, w // 2 + 4:w // 2 + 18].mean()
    assert up_left > 0.01 and low_right < -0.01           # highlight on the lit side, shadow on the far side
    assert up_left - low_right > 0.05


def test_more_depth_means_stronger_shading():
    b = bump()
    weak, strong = np.abs(relief(b, 0.2) - b).mean(), np.abs(relief(b, 0.9) - b).mean()
    assert strong > 2.5 * weak


def test_the_effect_is_steady_across_resolutions():
    def strength(h, w):
        b = bump(h, w)
        return np.abs(relief(b, 0.7) - b).max()
    small, big = strength(60, 100), strength(240, 400)
    assert 0.4 < small / big < 2.5


def test_bilinear_is_smooth_and_range_preserving():
    small = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    big = resize_bilinear(small, 8, 8)
    assert big.shape == (8, 8)
    assert big.min() >= 0.0 and big.max() <= 1.0
    assert len(np.unique(np.round(big, 3))) > 4