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
        assert 0.3 <= cfg.overlap_invert <= 1.0 and 0.0 <= cfg.pixel_decay <= 0.6


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
    assert a.gain == 0.4 and a.fullscreen is False and a.show_now_playing is True   # only the look changes


def test_randomize_across_everything_visits_every_version_and_fusions():
    cfg = Config()
    rng = np.random.default_rng(2)
    seen, fused = set(), 0
    for _ in range(200):
        randomize(cfg, rng)
        assert 1 <= len(cfg.versions) <= 3
        seen.update(cfg.versions)
        fused += len(cfg.versions) > 1
        for v in cfg.versions:
            assert cfg.elements[v]                                   # every chosen version keeps something on
    assert seen == {"v1", "v2", "v3", "v4", "v5", "v6", "v7"} and fused > 20


def test_randomize_within_keeps_the_chosen_versions():
    cfg = Config()
    cfg.versions = ["v2", "v5"]
    rng = np.random.default_rng(3)
    changed = 0
    for _ in range(30):
        before = {k: list(v) for k, v in cfg.elements.items()}
        randomize(cfg, rng, within=True)
        assert cfg.versions == ["v2", "v5"]
        changed += before["v2"] != cfg.elements["v2"] or before["v5"] != cfg.elements["v5"]
        assert cfg.elements["v7"] == before["v7"]                   # versions you didn't pick are left alone
    assert changed > 20


def test_the_button_changes_the_look():
    cfg = Config()
    before = Config()
    changed = 0
    rng = np.random.default_rng(3)
    for _ in range(20):
        randomize(cfg, rng)
        changed += cfg != before
    assert changed >= 19
