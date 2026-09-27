import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pytest

from visualizer import presets
from visualizer.config import Config


def test_the_default_preset_is_the_look_from_the_screenshots():
    cfg = Config()
    presets.apply(cfg, presets.DEFAULT_PRESET)
    assert cfg.versions == ["v1", "v4"]
    assert cfg.render_mode == "chars" and cfg.glyph_cell == 8 and cfg.bit_depth == 1
    assert cfg.glyph_sets == ["ascii"] and cfg.glyph_mapping == "brightness"
    assert (cfg.intensity, cfg.pixel_decay, cfg.overlap_invert) == (2.0, 0.11, 1.0)
    assert (cfg.mix_bass, cfg.mix_lowmid, cfg.mix_vocals, cfg.mix_highmid, cfg.mix_treble) == (2.0, 0.4, 1.9, 0.4, 2.0)
    assert cfg.elements["v1"] == ["spots", "glow", "drift"]


def test_a_preset_round_trips_exactly():
    cfg = Config()
    cfg.versions, cfg.intensity, cfg.mix_vocals = ["v2", "v6"], 0.7, 1.3
    cfg.elements["v2"] = ["orb", "depth"]
    cfg.render_mode, cfg.pixel_size = "pixels", 4
    other = Config()
    presets.apply(other, presets.snapshot(cfg))
    assert presets.snapshot(other) == presets.snapshot(cfg)


def test_share_codes_round_trip_and_survive_whitespace():
    cfg = Config()
    presets.apply(cfg, presets.DEFAULT_PRESET)
    code = presets.encode(presets.snapshot(cfg))
    assert code.startswith("AVP1.") and len(code) < 600
    wrapped = "here you go:\n  " + code[:40] + "\n" + code[40:] + "  "
    assert presets.decode(wrapped) == presets.snapshot(cfg)


@pytest.mark.parametrize("junk", ["", "hello", "AVP1.", "AVP1.!!!notbase64", "AVP1." + "QUJD"])
def test_bad_codes_are_refused_not_crashed_on(junk):
    with pytest.raises(ValueError):
        presets.decode(junk)


def test_applying_checks_every_value():
    cfg = Config()
    presets.apply(cfg, {"intensity": 99, "mix_bass": -3, "bit_depth": 7, "versions": ["v9", "v3"],
                        "glyph_sets": ["nope"], "elements": {"v3": ["bogus"], "v9": ["x"]}, "render_mode": "lasers",
                        "not_a_setting": 1, "pixel_size": "abc"})
    assert cfg.intensity == 2.0 and cfg.mix_bass == 0.0 and cfg.bit_depth == Config().bit_depth
    assert cfg.versions == ["v3"] and cfg.glyph_sets == Config().glyph_sets
    assert cfg.elements["v3"] == Config().elements["v3"] and cfg.render_mode == "pixels"


def test_window_settings_are_not_part_of_a_preset():
    cfg = Config()
    cfg.fullscreen = True
    presets.apply(cfg, presets.DEFAULT_PRESET)
    assert cfg.fullscreen is True and "fullscreen" not in presets.snapshot(cfg)


def test_store_saves_names_loads_and_deletes(tmp_path):
    store = presets.PresetStore(tmp_path / "p" / "presets.json")
    assert store.load() == []
    assert store.add({"intensity": 0.5}) == "Preset 1"
    assert store.add({"intensity": 1.5}) == "Preset 2"
    assert store.add({"intensity": 1.0}, "Mine") == "Mine"
    assert [p["name"] for p in store.load()] == ["Preset 1", "Preset 2", "Mine"]
    assert store.delete("Preset 1") and not store.delete("Preset 1")
    assert store.add({}) == "Preset 3"                              # never reuses a name that exists


def test_a_broken_or_unwritable_file_never_crashes(tmp_path):
    bad = tmp_path / "presets.json"
    bad.write_text("{not json")
    assert presets.PresetStore(bad).load() == []
    blocked = presets.PresetStore(tmp_path / "file" / "presets.json")
    (tmp_path / "file").write_text("I am a file, not a folder")
    assert blocked.add({}) is None


def manager(tmp_path, clip=""):
    cfg = Config()
    presets.apply(cfg, presets.DEFAULT_PRESET)
    box = {"clip": clip, "applied": 0}
    m = presets.PresetManager(cfg, on_applied=lambda: box.__setitem__("applied", box["applied"] + 1),
                              store=presets.PresetStore(tmp_path / "presets.json"),
                              clip_get=lambda: box["clip"], clip_put=lambda t: box.__setitem__("clip", t) or True)
    return cfg, m, box


def test_manager_saves_shares_pastes_and_deletes(tmp_path):
    cfg, m, box = manager(tmp_path)
    m.save()
    assert m.selected == "u:Preset 1" and "Saved" in m.message
    m.share()
    assert box["clip"].startswith("AVP1.") and "copied" in m.message
    code = box["clip"]
    m.choose("b:field")                                              # something else entirely
    assert cfg.versions == ["v7"] and box["applied"] == 1
    box["clip"] = code
    m.paste()                                                        # a friend's code brings the look back
    assert cfg.versions == ["v1", "v4"] and cfg.intensity == 2.0 and box["applied"] == 2
    assert [o[1] for o in m.options()][:2] == ["Default", "V7 Field (clean)"]
    assert m.can_delete()
    m.delete()
    assert m.selected == "b:default"


def test_pasting_something_that_isnt_a_code_explains_itself(tmp_path):
    cfg, m, box = manager(tmp_path, clip="just some text")
    before = presets.snapshot(cfg)
    m.paste()
    assert "doesn't hold a preset code" in m.message and presets.snapshot(cfg) == before


def test_the_list_shows_built_ins_then_your_newest(tmp_path):
    cfg, m, _ = manager(tmp_path)
    for _ in range(9):
        m.save()
    names = [o[1] for o in m.options()]
    assert len(names) == m.MAX_SHOWN and names[:2] == ["Default", "V7 Field (clean)"] and names[2] == "Preset 9"
