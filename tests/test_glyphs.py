import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame
import pytest

from visualizer.glyphs import BUILTIN_SHAPES, GlyphField, build_atlas, char_coverage, glyph_set, shape_coverage

pygame.init()
CELL = 12


def test_every_builtin_shape_draws_something_reasonable():
    for name in BUILTIN_SHAPES:
        cov = shape_coverage(name, CELL, CELL)
        assert cov.shape == (CELL, CELL)
        assert 0.03 < cov.mean() < 0.9, name
        assert cov.max() > 0.8, name


def test_shapes_are_different_from_each_other():
    covs = {n: shape_coverage(n, CELL, CELL) for n in ("circle", "square", "triangle", "diamond", "plus")}
    names = list(covs)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.allclose(covs[a], covs[b]), (a, b)


def test_characters_render_and_whitespace_does_not():
    assert char_coverage("A", "", CELL, CELL) is not None
    assert char_coverage(" ", "", CELL, CELL) is None
    assert char_coverage("", "", CELL, CELL) is None


def test_typed_characters_join_the_glyph_set_once_each():
    base = len(glyph_set(["circle", "square"], "", "", CELL))
    assert base == 2
    assert len(glyph_set(["circle", "square"], "ab ba", "", CELL)) == base + 2  # a, b (deduped, space skipped)


def test_empty_selection_still_gives_something_to_draw():
    assert len(glyph_set([], "", "", CELL)) == 1


def test_symbol_fonts_are_usable_when_installed():
    if "wingdings" not in pygame.font.get_fonts():
        pytest.skip("no Wingdings on this machine")
    assert len(glyph_set([], "lmn", "wingdings", CELL)) == 3


def test_atlas_levels_get_brighter_and_bigger():
    atlas = build_atlas([shape_coverage("circle", CELL, CELL)], 4, CELL)
    assert atlas.shape == (1, 4, CELL, CELL)
    assert atlas[0, 0].sum() == 0                                   # level 0 is blank
    sums = [int(atlas[0, l].sum()) for l in range(1, 4)]
    assert sums[0] < sums[1] < sums[2]


def _field(mapping_shapes=("circle", "square", "triangle", "diamond", "plus"), bits=2, chars=""):
    gf = GlyphField(seed=1)
    gf.configure(list(mapping_shapes), chars, "", CELL, bits)
    return gf


def test_render_shape_and_blank_and_full():
    gf = _field()
    rows, cols = 9, 14
    assert gf.render(np.zeros((rows, cols), np.float32)).max() == 0
    img = gf.render(np.ones((rows, cols), np.float32))
    assert img.shape == (rows * CELL, cols * CELL) and img.dtype == np.uint8
    assert (img > 0).mean() > 0.15


def test_random_mapping_is_stable_until_reshuffled():
    inten = np.full((10, 10), 0.9, np.float32)
    gf = _field()
    a, b = gf.render(inten).copy(), gf.render(inten).copy()
    assert np.array_equal(a, b)
    gf.reshuffle()
    assert not np.array_equal(a, gf.render(inten))


def test_brightness_mapping_gives_denser_glyphs_to_brighter_cells():
    gf = _field(bits=1)
    dark = gf.render(np.full((6, 6), 0.34, np.float32), "brightness")
    bright = gf.render(np.full((6, 6), 0.99, np.float32), "brightness")
    # 1-bit: a cell is either blank or its full glyph, so compare ink per lit cell
    def ink_per_lit_cell(img):
        cells = img.reshape(6, CELL, 6, CELL).transpose(0, 2, 1, 3).reshape(36, -1).sum(axis=1)
        lit = cells[cells > 0]
        return lit.mean() if len(lit) else 0
    assert ink_per_lit_cell(bright) > ink_per_lit_cell(dark)


def test_settings_change_rebuilds_the_atlas_only_when_needed():
    gf = _field()
    atlas = gf._atlas
    gf.configure(["circle", "square", "triangle", "diamond", "plus"], "", "", CELL, 2)
    assert gf._atlas is atlas                       # nothing changed: reused
    gf.configure(["circle"], "", "", CELL, 2)
    assert gf._atlas is not atlas and gf.glyph_count == 1


def test_bit_depth_sets_the_number_of_brightness_levels():
    assert _field(bits=1)._atlas.shape[1] == 2
    assert _field(bits=3)._atlas.shape[1] == 8
