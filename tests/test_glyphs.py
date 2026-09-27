import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from visualizer.glyphs import (ASCII_CHARS, SHAPE_NAMES, SYMBOL_NAMES, GlyphField, build_atlas, char_coverage, glyph_set,
                               shape_coverage, symbol_coverage)

pygame.init()
CELL = 16


def test_every_builtin_shape_draws_something_reasonable():
    for name in SHAPE_NAMES:
        cov = shape_coverage(name, CELL, CELL)
        assert cov.shape == (CELL, CELL)
        assert 0.03 < cov.mean() < 0.9, name
        assert cov.max() > 0.8, name


def test_every_symbol_draws_and_they_are_all_different():
    covs = {n: symbol_coverage(n, CELL, CELL) for n in SYMBOL_NAMES}
    for name, cov in covs.items():
        assert cov.shape == (CELL, CELL)
        assert 0.04 < cov.mean() < 0.85, name
        assert cov.max() > 0.8, name
    names = list(covs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(covs[a], covs[b]), (a, b)


def test_shapes_are_different_from_each_other():
    covs = {n: shape_coverage(n, CELL, CELL) for n in SHAPE_NAMES}
    names = list(covs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(covs[a], covs[b]), (a, b)


def test_characters_render_and_whitespace_does_not():
    assert char_coverage("A", "", CELL, CELL) is not None
    assert char_coverage(" ", "", CELL, CELL) is None
    assert char_coverage("", "", CELL, CELL) is None


def test_the_four_sets_have_the_expected_glyphs():
    assert len({g.tobytes() for g in glyph_set(["shapes"], CELL)}) == len(SHAPE_NAMES)
    assert len({g.tobytes() for g in glyph_set(["symbols"], CELL)}) == len(SYMBOL_NAMES)
    assert len({g.tobytes() for g in glyph_set(["ascii"], CELL)}) >= len(ASCII_CHARS) - 1
    assert len({g.tobytes() for g in glyph_set(["binary"], CELL)}) == 2


def test_small_sets_are_repeated_so_each_set_gets_a_fair_share():
    binary = glyph_set(["binary"], CELL)
    assert len(binary) >= 6                                  # 0 and 1, repeated
    mixed = glyph_set(["shapes", "binary"], CELL)
    zeros_ones = sum(1 for g in mixed if any(np.array_equal(g, b) for b in binary[:2]))
    assert zeros_ones >= 0.3 * len(mixed)                    # binary is not drowned out by shapes


def test_sets_combine():
    both = glyph_set(["shapes", "symbols"], CELL)
    assert len(both) == len(SHAPE_NAMES) + len(SYMBOL_NAMES)


def test_no_sets_still_gives_something_to_draw():
    assert len(glyph_set([], CELL)) == 1
    assert len(glyph_set(["nonsense"], CELL)) == 1


def test_atlas_levels_get_brighter_and_bigger():
    atlas = build_atlas([shape_coverage("circle", CELL, CELL)], 4, CELL)
    assert atlas.shape == (1, 4, CELL, CELL)
    assert atlas[0, 0].sum() == 0                                   # level 0 is blank
    sums = [int(atlas[0, l].sum()) for l in range(1, 4)]
    assert sums[0] < sums[1] < sums[2]


def _field(sets=("shapes",), bits=2):
    gf = GlyphField(seed=1)
    gf.configure(list(sets), CELL, bits)
    return gf


def test_render_shape_and_blank_and_full():
    gf = _field()
    rows, cols = 9, 14
    assert gf.render(np.zeros((rows, cols), np.float32)).max() == 0
    img = gf.render(np.ones((rows, cols), np.float32))
    assert img.shape == (rows * CELL, cols * CELL) and img.dtype == np.uint8
    assert (img > 0).mean() > 0.1


def test_random_mapping_is_stable_until_reshuffled():
    inten = np.full((10, 10), 0.9, np.float32)
    gf = _field()
    a, b = gf.render(inten).copy(), gf.render(inten).copy()
    assert np.array_equal(a, b)
    gf.reshuffle()
    assert not np.array_equal(a, gf.render(inten))


def test_brightness_mapping_gives_denser_glyphs_to_brighter_cells():
    gf = _field(sets=("ascii",), bits=1)
    dark = gf.render(np.full((6, 6), 0.3, np.float32), "brightness")
    bright = gf.render(np.full((6, 6), 0.99, np.float32), "brightness")

    def ink_per_lit_cell(img):
        cells = img.reshape(6, CELL, 6, CELL).transpose(0, 2, 1, 3).reshape(36, -1).sum(axis=1)
        lit = cells[cells > 0]
        return lit.mean() if len(lit) else 0
    assert ink_per_lit_cell(bright) > ink_per_lit_cell(dark)


def test_settings_change_rebuilds_the_atlas_only_when_needed():
    gf = _field()
    atlas = gf._atlas
    gf.configure(["shapes"], CELL, 2)
    assert gf._atlas is atlas                       # nothing changed: reused
    gf.configure(["binary"], CELL, 2)
    assert gf._atlas is not atlas


def test_bit_depth_sets_the_number_of_brightness_levels():
    assert _field(bits=1)._atlas.shape[1] == 2
    assert _field(bits=3)._atlas.shape[1] == 8
