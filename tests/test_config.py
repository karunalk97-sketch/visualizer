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
    assert cfg.overlap_invert > 0
    assert not hasattr(cfg, "waves") and not hasattr(cfg, "show_shapes")           # the wave layer and shape toggle were removed
    assert not hasattr(cfg, "depth")                                               # the 3D relief lighting was removed
    assert set(cfg.glyph_sets) <= {k for k, _ in GLYPH_SETS}
    assert not hasattr(cfg, "glyph_font") and not hasattr(cfg, "glyph_chars")     # custom fonts/text were removed


def test_sensitivity_is_fixed_and_intensity_decay_versions_default_sensibly():
    cfg = Config()
    assert cfg.gain == 0.4               # fixed, calibrated on real music
    assert cfg.intensity == 1.0          # normal: subtle below, jarring above
    assert cfg.pixel_decay == 0.0
    assert cfg.versions == ["v7"] and set(cfg.elements) == {"v1", "v2", "v3", "v4", "v5", "v6", "v7"}
    a, b = Config(), Config()
    a.elements["v7"].remove("ink")
    assert "ink" in b.elements["v7"]     # each session starts from its own fresh defaults
