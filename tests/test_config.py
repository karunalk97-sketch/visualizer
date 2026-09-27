from visualizer.config import GLYPH_SETS, Config


def test_settings_are_session_only():
    """Nothing is saved between launches: there is no save/load API at all."""
    cfg = Config()
    assert not hasattr(cfg, "save") and not hasattr(Config, "load")
    import visualizer.config as config_module
    assert not hasattr(config_module, "default_config_path")


def test_every_launch_starts_from_the_same_defaults():
    a, b = Config(), Config()
    assert a == b
    a.glyph_sets.append("ascii")
    assert Config().glyph_sets == ["shapes"]           # the default list is not shared between sessions


def test_defaults_are_sensible():
    cfg = Config()
    assert cfg.render_mode == "pixels" and cfg.pixel_size == 3 and cfg.bit_depth == 1
    assert cfg.show_shapes and cfg.waves and cfg.depth > 0
    assert set(cfg.glyph_sets) <= {k for k, _ in GLYPH_SETS}
    assert not hasattr(cfg, "glyph_font") and not hasattr(cfg, "glyph_chars")     # custom fonts/text were removed
