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


def test_three_tabs_and_clicking_them_switches():
    cfg, panel, _ = opened()
    assert [k for k, _ in TABS] == ["look", "layers", "chars"]
    for k, _ in TABS:
        click(panel, panel._header["tab:" + k].center)
        assert panel.tab == k


def test_arrow_keys_switch_tabs_and_wrap():
    cfg, panel, _ = opened()
    key(panel, pygame.K_RIGHT)
    assert panel.tab == "layers"
    key(panel, pygame.K_RIGHT); key(panel, pygame.K_RIGHT)
    assert panel.tab == "look"                         # wrapped around
    key(panel, pygame.K_LEFT)
    assert panel.tab == "chars"


def test_each_tab_shows_its_own_controls():
    cfg, panel, _ = opened("look")
    assert {"Draw with", "Bit depth", "3D depth", "Sensitivity", "Fullscreen"} <= set(labels(panel))
    panel.set_tab("layers")
    assert {"Shapes", "Waves (sea foam)", "Wave strength", "Wave softness", "How often waves come"} <= set(labels(panel))
    panel.set_tab("chars")
    assert {"Character sets", "Pick glyphs", "Character size"} <= set(labels(panel))


def test_the_old_font_and_text_controls_are_gone():
    cfg, panel, _ = opened()
    every = []
    for k, _ in TABS:
        panel.set_tab(k)
        every += labels(panel)
    assert not any("font" in l.lower() or "characters (paste" in l.lower() or "type" in l.lower() for l in every)


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
    cfg, panel, calls = opened("layers")
    assert cfg.waves
    click(panel, row(panel, "Waves (sea foam)").rect.center)
    assert cfg.waves is False and "waves" in calls["changed"]
    click(panel, row(panel, "Waves (sea foam)").rect.center)
    assert cfg.waves is True


def test_shapes_and_waves_switch_independently():
    cfg, panel, _ = opened("layers")
    click(panel, row(panel, "Shapes").rect.center)
    assert cfg.show_shapes is False and cfg.waves is True


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
    cfg, panel, _ = opened("layers")
    track = row(panel, "Wave softness").parts[0][0]
    click(panel, (track.x, track.centery))
    assert cfg.wave_softness == 0.0 and panel._drag is not None
    move(panel, (track.x + track.width // 2, track.centery + 40))     # vertical wobble doesn't drop the drag
    assert 0.45 < cfg.wave_softness < 0.55
    move(panel, (track.right + 300, track.centery))                    # dragging past the end clamps
    assert cfg.wave_softness == 1.0
    release(panel, (track.right, track.centery))
    assert panel._drag is None
    move(panel, (track.x, track.centery))
    assert cfg.wave_softness == 1.0                                    # released: no longer follows


def test_slider_can_be_grabbed_anywhere_on_its_row():
    cfg, panel, _ = opened("layers")
    r = row(panel, "Wave strength")
    click(panel, (r.rect.centerx, r.rect.y + 3))        # near the label, well above the track
    assert 0.4 < cfg.wave_strength < 0.7


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
    cfg, panel, _ = opened("layers")
    r = row(panel, "Wave strength")
    assert r.rect.height >= 40                                             # the whole row is grabbable
    cfg2, panel2, _ = opened("look")
    for rect, _ in row(panel2, "Pixel size").parts:
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
