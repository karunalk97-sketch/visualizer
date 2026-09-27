import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame
import pytest

from visualizer.config import GLYPH_CELL_STEPS, PIXEL_STEPS, Config
from visualizer.settings_ui import TABS, SettingsPanel

pygame.init()
pygame.display.set_mode((1280, 720))
W, H = 1280, 690


def make():
    cfg = Config()
    calls = {"changed": [], "reshuffled": 0, "randomized": 0}
    panel = SettingsPanel(cfg, calls["changed"].append,
                          lambda: calls.__setitem__("reshuffled", calls["reshuffled"] + 1),
                          lambda: calls.__setitem__("randomized", calls["randomized"] + 1))
    return cfg, panel, calls


def opened(tab="look"):
    cfg, panel, calls = make()
    panel.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0, unicode=""), W, H)
    panel.set_tab(tab)
    panel.layout(W, H)
    return cfg, panel, calls


def click(panel, pos):
    return panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1), W, H)


def release(panel, pos):
    return panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=pos, button=1), W, H)


def move(panel, pos):
    return panel.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=pos, rel=(0, 0), buttons=(0, 0, 0)), W, H)


def key(panel, k):
    return panel.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode=""), W, H)


def row(panel, label):
    return next(r for r in panel.layout(W, H) if r.label == label)


def labels(panel):
    return [r.label for r in panel.layout(W, H)]


# -- open / close / tabs ------------------------------------------------------------

def test_tab_key_opens_and_closes_and_escape_closes():
    cfg, panel, _ = make()
    assert not panel.visible
    assert key(panel, pygame.K_TAB) and panel.visible
    assert key(panel, pygame.K_ESCAPE) and not panel.visible
    assert not key(panel, pygame.K_ESCAPE)            # closed: not consumed, main can use it


def test_six_tabs_and_clicking_them_switches():
    cfg, panel, _ = opened()
    assert [k for k, _ in TABS] == ["look", "versions", "mix", "chars", "audio", "presets"]
    for k, _ in TABS:
        click(panel, panel._header["tab:" + k].center)
        assert panel.tab == k


def test_arrow_keys_switch_tabs_and_wrap():
    cfg, panel, _ = opened()
    for expected in ("versions", "mix", "chars", "audio", "presets", "look"):
        key(panel, pygame.K_RIGHT)
        assert panel.tab == expected
    key(panel, pygame.K_LEFT)
    assert panel.tab == "presets"


def test_presets_tab_applies_saves_shares_pastes_and_deletes(tmp_path):
    from visualizer import presets
    cfg = Config()
    clip = {"t": ""}
    changed = []
    mgr = presets.PresetManager(cfg, store=presets.PresetStore(tmp_path / "p.json"),
                                clip_get=lambda: clip["t"], clip_put=lambda t: clip.__setitem__("t", t) or True)
    panel = SettingsPanel(cfg, changed.append, lambda: None, lambda: None, presets=mgr)
    panel.visible = True
    panel.set_tab("presets")
    click(panel, next(r for r, p in row(panel, "Presets  (click one to apply it)").parts if p == "b:default").center)
    assert cfg.versions == ["v1", "v4"] and "preset" in changed
    click(panel, row(panel, "Save the current look as a preset").rect.center)
    assert "Preset 1" in [label for _, label, _ in mgr.options()]
    click(panel, row(panel, "Copy a share code").rect.center)
    assert clip["t"].startswith("AVP1.")
    assert "Delete this preset" in labels(panel)
    click(panel, row(panel, "Delete this preset").rect.center)
    assert "Delete this preset" not in labels(panel)
    click(panel, next(r for r, p in row(panel, "Presets  (click one to apply it)").parts if p == "b:field").center)
    assert cfg.versions == ["v7"]
    click(panel, row(panel, "Paste a shared preset").rect.center)
    assert cfg.versions == ["v1", "v4"]                                   # the copied code came back


def test_each_tab_shows_its_own_controls():
    cfg, panel, _ = opened("look")
    assert {"Draw with", "Bit depth", "Decay", "Overlap inversion", "Intensity  (subtle - jarring)", "Fullscreen"} <= set(labels(panel))
    assert "3D depth" not in labels(panel) and "Sensitivity" not in labels(panel)   # Sensitivity is fixed now
    panel.set_tab("chars")
    assert {"Character sets", "Pick glyphs", "Character size"} <= set(labels(panel))
    panel.set_tab("mix")
    assert {"Bass / kick  (ink)", "Low mids  (orbs)", "Vocals / melody  (rings)",
            "Snare / high mids  (stars)", "Hats / treble  (sand)"} <= set(labels(panel))


def chip(panel, row_label, key):
    return next(rect for rect, p in row(panel, row_label).parts if p == key).center


def test_versions_tab_picks_and_fuses_versions_and_keeps_at_least_one():
    cfg, panel, calls = opened("versions")
    assert cfg.versions == ["v7"]
    vrow = "Versions  (pick several to fuse them)"
    click(panel, chip(panel, vrow, "v1"))
    click(panel, chip(panel, vrow, "v4"))
    assert cfg.versions == ["v1", "v4", "v7"]                     # fused, in a stable order
    for k in ("v1", "v4", "v7"):
        click(panel, chip(panel, vrow, k))
    assert len(cfg.versions) == 1                                 # the last one can't be switched off
    assert "versions" in calls["changed"]


def test_versions_tab_edits_the_elements_of_each_chosen_version():
    cfg, panel, calls = opened("versions")
    click(panel, chip(panel, "Versions  (pick several to fuse them)", "v4"))    # adding a version shows its elements
    assert "V4 Weather" in labels(panel)
    click(panel, chip(panel, "V4 Weather", "terrain"))
    assert "terrain" not in cfg.elements["v4"] and "elements" in calls["changed"]
    click(panel, next(rect for rect, p in row(panel, "Elements of").parts if p == "v7").center)
    assert "V7 Field" in labels(panel)
    for e in list(cfg.elements["v7"]):
        click(panel, chip(panel, "V7 Field", e))
    assert len(cfg.elements["v7"]) == 1                           # a version keeps at least one element


def test_randomize_within_calls_its_own_callback():
    cfg = Config()
    calls = []
    panel = SettingsPanel(cfg, lambda n: None, lambda: None, lambda: calls.append("all"),
                          on_randomize_within=lambda: calls.append("within"))
    panel.visible = True
    panel.set_tab("versions")
    panel.layout(W, H)
    click(panel, row(panel, "Randomize within these versions").rect.center)
    click(panel, panel._header["randomize"].center)
    assert calls == ["within", "all"]


def test_mix_sliders_set_the_mix():
    cfg, panel, _ = opened("mix")
    track = row(panel, "Bass / kick  (ink)").parts[0][0]
    click(panel, (track.right - 1, track.centery))
    assert cfg.mix_bass > 1.95
    release(panel, (track.right - 1, track.centery))
    click(panel, (track.x, track.centery))
    assert cfg.mix_bass == 0.0


def test_no_wave_or_font_or_text_controls_remain():
    cfg, panel, _ = opened()
    every = []
    for k, _ in TABS:
        panel.set_tab(k)
        every += labels(panel)
    joined = " ".join(every).lower()
    assert "wave" not in joined and "foam" not in joined
    assert "font" not in joined and "type" not in joined and "paste" not in joined
    assert not any(hasattr(cfg, a) for a in ("waves", "wave_strength", "wave_softness", "wave_rate", "show_shapes"))


def test_close_button_and_clicks_outside():
    cfg, panel, _ = opened()
    assert click(panel, panel._header["close"].center) and not panel.visible
    cfg, panel, _ = opened()
    assert not click(panel, (50, 50))                  # outside the panel: passes through to the app
    assert click(panel, (W - 3, 3))                    # inside: consumed even on empty space


def test_closed_panel_consumes_nothing():
    cfg, panel, _ = make()
    assert not click(panel, (W - 5, 5))
    assert not move(panel, (W - 5, 5))


# -- controls -----------------------------------------------------------------------

def test_toggle_flips_the_setting_and_reports_it():
    cfg, panel, calls = opened("look")
    assert cfg.show_now_playing
    click(panel, row(panel, "Track and status bar text").rect.center)
    assert cfg.show_now_playing is False and "show_now_playing" in calls["changed"]
    click(panel, row(panel, "Track and status bar text").rect.center)
    assert cfg.show_now_playing is True


def test_a_toggle_responds_anywhere_on_its_row():
    cfg, panel, _ = opened("look")
    r = row(panel, "Fullscreen")
    click(panel, (r.rect.x + 5, r.rect.centery))         # on the label, not just the switch
    assert cfg.fullscreen is True


def test_pixel_stepper_moves_through_the_allowed_sizes_and_clamps():
    cfg, panel, _ = opened("look")
    assert cfg.pixel_size == 3
    click(panel, row(panel, "Pixel size").parts[1][0].center)
    assert cfg.pixel_size == PIXEL_STEPS[PIXEL_STEPS.index(3) + 1]
    for _ in range(8):
        click(panel, row(panel, "Pixel size").parts[0][0].center)
    assert cfg.pixel_size == PIXEL_STEPS[0]


def test_bit_depth_stepper_stays_in_range():
    cfg, panel, _ = opened("look")
    for _ in range(6):
        click(panel, row(panel, "Bit depth").parts[1][0].center)
    assert cfg.bit_depth == 4


def test_slider_click_and_drag_follow_the_mouse():
    cfg, panel, _ = opened("look")
    track = row(panel, "Decay").parts[0][0]
    click(panel, (track.x, track.centery))
    assert cfg.pixel_decay == 0.0 and panel._drag is not None
    move(panel, (track.x + track.width // 2, track.centery + 40))     # vertical wobble doesn't drop the drag
    assert 0.4 < cfg.pixel_decay < 0.5
    move(panel, (track.right + 300, track.centery))                    # dragging past the end clamps
    assert cfg.pixel_decay == 0.9
    release(panel, (track.right, track.centery))
    assert panel._drag is None
    move(panel, (track.x, track.centery))
    assert cfg.pixel_decay == 0.9                                      # released: no longer follows


def test_slider_can_be_grabbed_anywhere_on_its_row():
    cfg, panel, _ = opened("look")
    r = row(panel, "Overlap inversion")
    click(panel, (r.rect.centerx, r.rect.y + 3))        # near the label, well above the track
    assert 0.4 < cfg.overlap_invert < 0.6


def test_draw_with_choice_and_the_size_stepper_follows_the_mode():
    cfg, panel, _ = opened("look")
    assert "Pixel size" in labels(panel) and "Character size" not in labels(panel)
    r = row(panel, "Draw with")
    click(panel, next(rect for rect, p in r.parts if p == "chars").center)
    assert cfg.render_mode == "chars"
    assert "Character size" in labels(panel) and "Pixel size" not in labels(panel)


def test_character_sets_toggle_and_at_least_one_stays_on():
    cfg, panel, calls = opened("chars")
    assert cfg.glyph_sets == ["shapes"]
    chip = lambda key: next(rect for rect, p in row(panel, "Character sets").parts if p == key).center   # noqa: E731
    click(panel, chip("ascii"))
    click(panel, chip("binary"))
    assert cfg.glyph_sets == ["shapes", "ascii", "binary"]          # stable order
    click(panel, chip("shapes"))
    assert cfg.glyph_sets == ["ascii", "binary"]
    click(panel, chip("ascii")); click(panel, chip("binary"))
    assert len(cfg.glyph_sets) == 1                                  # can't switch the last one off
    assert "glyph_sets" in calls["changed"]


def test_mapping_choice_and_character_size():
    cfg, panel, _ = opened("chars")
    r = row(panel, "Pick glyphs")
    click(panel, next(rect for rect, p in r.parts if p == "brightness").center)
    assert cfg.glyph_mapping == "brightness"
    click(panel, row(panel, "Character size").parts[1][0].center)
    assert cfg.glyph_cell == GLYPH_CELL_STEPS[GLYPH_CELL_STEPS.index(12) + 1]


def test_characters_tab_explains_itself_while_drawing_with_pixels():
    cfg, panel, _ = opened("chars")
    assert any("Characters on the Look tab" in l for l in labels(panel))
    cfg.render_mode = "chars"
    assert not any("Characters on the Look tab" in l for l in labels(panel))


def test_randomize_and_reshuffle_buttons_call_back():
    cfg, panel, calls = opened("look")
    click(panel, panel._header["randomize"].center)
    assert calls["randomized"] == 1
    click(panel, row(panel, "Reshuffle the layout now").rect.center)
    assert calls["reshuffled"] == 1


def test_randomize_is_reachable_from_every_tab():
    cfg, panel, calls = opened()
    for k, _ in TABS:
        panel.set_tab(k)
        panel.layout(W, H)
        click(panel, panel._header["randomize"].center)
    assert calls["randomized"] == len(TABS)


# -- responsiveness / layout ----------------------------------------------------------

def test_hover_tracks_the_row_under_the_mouse():
    cfg, panel, _ = opened("look")
    r = row(panel, "Fullscreen")
    move(panel, r.rect.center)
    assert panel.hover is not None and panel.hover.label == "Fullscreen"
    move(panel, (50, 50))
    assert panel.hover is None


def test_interactive_areas_are_reported_for_the_hand_cursor():
    cfg, panel, _ = opened("look")
    assert panel.interactive_at(row(panel, "Fullscreen").rect.center)
    assert panel.interactive_at(panel._header["randomize"].center)
    assert not panel.interactive_at((50, 50))


@pytest.mark.parametrize("height", [420, 520, 690, 1000, 1300])
def test_everything_fits_on_screen_without_scrolling_and_targets_stay_big(height):
    cfg, panel, _ = make()
    cfg.render_mode = "chars"
    for k, _ in TABS:
        panel.set_tab(k)
        rows = panel.layout(1600, height)
        assert rows
        top = panel._header["randomize"].top
        assert max(r.rect.bottom for r in rows) <= top, (k, height)       # nothing runs under the Randomize button
        for r in rows:
            for rect, _ in r.parts:
                assert rect.height >= (22 if height >= 690 else 17), (k, height, r.label)   # every button is easy to hit
        assert panel._header["randomize"].bottom <= height


def test_slider_and_pill_targets_are_generous_at_the_default_size():
    cfg, panel, _ = opened("look")
    r = row(panel, "Intensity  (subtle - jarring)")
    assert r.rect.height >= 40                                             # the whole row is grabbable
    for rect, _ in row(panel, "Pixel size").parts:
        assert rect.width >= 32 and rect.height >= 32


def test_drawing_the_panel_is_cheap_and_reuses_cached_text():
    cfg, panel, _ = opened("look")
    surface = pygame.display.get_surface()
    panel.draw(surface, 1280, 690)
    n = len(panel._text)
    for _ in range(30):
        panel.draw(surface, 1280, 690)
    assert len(panel._text) == n                                            # no new text surfaces once warm
    for k, _ in TABS:
        panel.set_tab(k)
        panel.draw(surface, 1280, 690)                                       # every tab draws without errors


# -- the Audio tab ------------------------------------------------------------------

class FakeAudio:
    def __init__(self, apps=(), supported=True, notice=""):
        self.selected = "system"
        self.supported = supported
        self.notice = notice
        self.refreshes = 0
        self.chosen = []
        self._apps = list(apps)

    def options(self):
        return [("system", "All system audio", "")] + [("app:" + k, label, note) for k, label, note in self._apps]

    def select(self, key):
        self.chosen.append(key)
        self.selected = key

    def request_refresh(self):
        self.refreshes += 1


def make_audio(apps=(("spotify", "Spotify", "playing"), ("chrome", "Chrome", "silent")), **kw):
    cfg = Config()
    audio = FakeAudio(apps, **kw)
    panel = SettingsPanel(cfg, lambda name: changes.append(name), lambda: None, lambda: None, audio=audio)
    changes = []
    panel.changes = changes
    panel.handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB, mod=0, unicode=""), W, H)
    panel.set_tab("audio")
    panel.layout(W, H)
    return audio, panel


def test_the_audio_tab_lists_system_audio_first_then_the_apps():
    audio, panel = make_audio()
    r = row(panel, "Listen to")
    assert [key for _, key in r.parts] == ["system", "app:spotify", "app:chrome"]


def test_clicking_an_app_selects_it_and_reports_the_change():
    audio, panel = make_audio()
    r = row(panel, "Listen to")
    click(panel, next(rect for rect, key in r.parts if key == "app:spotify").center)
    assert audio.chosen == ["app:spotify"] and audio.selected == "app:spotify"
    assert "audio_source" in panel.changes
    click(panel, next(rect for rect, key in row(panel, "Listen to").parts if key == "system").center)
    assert audio.selected == "system"


def test_the_list_refreshes_while_the_tab_is_open_and_not_otherwise():
    audio, panel = make_audio()
    n = audio.refreshes
    assert n >= 1
    panel.layout(W, H); panel.layout(W, H)
    assert audio.refreshes == n + 2                        # asked every frame (the router rate-limits it)
    panel.set_tab("look")
    panel.layout(W, H)
    assert audio.refreshes == n + 2                        # not asked on other tabs


def test_the_audio_tab_explains_when_only_system_audio_is_possible():
    audio, panel = make_audio(supported=False)
    assert any("Windows 10" in l for l in labels(panel))


def test_a_capture_problem_is_shown_to_the_user():
    audio, panel = make_audio(notice="Couldn't capture Spotify (access denied); using all system audio.")
    assert any("Couldn't capture Spotify" in l for l in labels(panel))


def test_an_empty_list_still_shows_system_audio_and_a_hint():
    audio, panel = make_audio(apps=())
    assert [key for _, key in row(panel, "Listen to").parts] == ["system"]
    assert any("Start playing" in l for l in labels(panel))


def test_without_an_audio_router_the_tab_says_so_instead_of_crashing():
    cfg, panel, _ = opened("audio")
    assert any("isn't available" in l for l in labels(panel))
    panel.draw(pygame.display.get_surface(), W, H)


def test_long_app_lists_are_capped_and_everything_still_fits():
    many = [(f"app{i}", f"App number {i}", "playing" if i % 2 else "silent") for i in range(12)]
    for height in (420, 520, 690, 1000):
        audio, panel = make_audio(apps=many, notice="x " * 60)
        panel.layout(1600, height)
        rows = panel.layout(1600, height)
        top = panel._header["randomize"].top
        assert max(r.rect.bottom for r in rows) <= top, height
        assert len(row(panel, "Listen to").parts) <= 6         # capped: system + five apps
        for r in rows:
            for rect, _ in r.parts:
                assert rect.height >= 14


def test_the_audio_tab_draws_with_selection_notes_and_wrapped_text():
    audio, panel = make_audio(notice="A fairly long explanation that has to wrap over several lines in the narrow panel.")
    audio.selected = "app:spotify"
    surface = pygame.display.get_surface()
    panel.draw(surface, W, H)
    n = len(panel._text)
    for _ in range(10):
        panel.draw(surface, W, H)
    assert len(panel._text) == n                                 # cached: no new text surfaces once warm