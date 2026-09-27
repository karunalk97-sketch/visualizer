import numpy as np

from visualizer.config import GLYPH_SETS, PIXEL_STEPS, Config
from visualizer.randomizer import randomize

KEYS = [k for k, _ in GLYPH_SETS]


def test_every_result_is_a_valid_watchable_combination():
    cfg = Config()
    rng = np.random.default_rng(0)
    for _ in range(400):
        randomize(cfg, rng)
        assert cfg.render_mode in ("pixels", "chars")
        assert cfg.bit_depth in (1, 2, 3)
        assert cfg.pixel_size in PIXEL_STEPS
        assert 6 <= cfg.glyph_cell <= 40
        assert cfg.glyph_sets and set(cfg.glyph_sets) <= set(KEYS)
        assert cfg.glyph_sets == [k for k in KEYS if k in cfg.glyph_sets]        # stable order, no duplicates
        assert cfg.glyph_mapping in ("random", "brightness")
        assert 0.3 <= cfg.overlap_invert <= 1.0 and 0.1 <= cfg.depth <= 0.8


def test_it_actually_varies_what_you_get():
    cfg = Config()
    rng = np.random.default_rng(1)
    seen = {"modes": set(), "sets": set(), "bits": set(), "mapping": set()}
    for _ in range(300):
        randomize(cfg, rng)
        seen["modes"].add(cfg.render_mode)
        seen["sets"].add(tuple(cfg.glyph_sets))
        seen["bits"].add(cfg.bit_depth)
        seen["mapping"].add(cfg.glyph_mapping)
    assert seen["modes"] == {"pixels", "chars"}
    assert len(seen["sets"]) >= 8
    assert seen["bits"] == {1, 2, 3} and seen["mapping"] == {"random", "brightness"}


def test_a_seed_gives_a_repeatable_look_and_leaves_other_settings_alone():
    a, b = Config(), Config()
    randomize(a, np.random.default_rng(5))
    randomize(b, np.random.default_rng(5))
    assert a == b
    assert a.gain == 1.0 and a.fullscreen is False and a.show_now_playing is True   # only the look changes


def test_the_button_changes_the_look():
    cfg = Config()
    before = Config()
    changed = 0
    rng = np.random.default_rng(3)
    for _ in range(20):
        randomize(cfg, rng)
        changed += cfg != before
    assert changed >= 19
