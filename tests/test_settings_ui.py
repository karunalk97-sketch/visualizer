import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import pygame

from visualizer.config import GLYPH_CELL_STEPS, PIXEL_STEPS, Config
from visualizer.settings_ui import SettingsPanel

pygame.init()
pygame.display.set_mode((1280, 720))
W, H = 1280, 690


def make():
    cfg = Config()
    calls = {"changed": [], "reshuffled": 0, "closed": 0}
    panel = SettingsPanel(cfg, calls["changed"].append, lambda: calls.__setitem__("reshuffled", calls["reshuffled"] + 1),
                          on_close=lambda: calls.__setitem__("closed", calls["closed"] + 1))
    return cfg, panel, calls


def click(panel, pos):
    return panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, pos=pos, button=1), W, H)


def row(panel, label):
    return next(r for r in panel.layout(W, H) if r.label == label)


def key(panel, k, mod=0):
    return panel.handle_event(pygame.event.Event(pygame.KEYDOWN, key=k, mod=mod, unicode=""), W, H)


def text(panel, s):
    return panel.handle_event(pygame.event.Event(pygame.TEXTINPUT, text=s), W, H)


def opened():
    cfg, panel, calls = make()
    key(panel, pygame.K_TAB)
    return cfg, panel, calls


def test_tab_opens_and_closes_and_close_saves():
    cfg, panel, calls = make()
    assert not panel.visible
    assert key(panel, pygame.K_TAB) and panel.visible
    assert key(panel, pygame.K_TAB) and not panel.visible
    assert calls["closed"] == 1


def test_escape_closes_the_open_panel_but_is_ignored_when_closed():
    cfg, panel, calls = opened()
    assert key(panel, pygame.K_ESCAPE) and not panel.visible
    assert not key(panel, pygame.K_ESCAPE)          # closed: not consumed, main can use it


def test_toggle_flips_the_setting_and_reports_it():
    cfg, panel, calls = opened()
    assert cfg.waves
    r = row(panel, "Waves")
    assert click(panel, r.rect.center)
    assert cfg.waves is False and "waves" in calls["changed"]
    click(panel, row(panel, "Waves").rect.center)
    assert cfg.waves is True


def test_shapes_and_waves_can_be_switched_independently():
    cfg, panel, _ = opened()
    click(panel, row(panel, "Shapes").rect.center)
    assert cfg.show_shapes is False and cfg.waves is True


def test_stepper_moves_through_the_allowed_pixel_sizes():
    cfg, panel, _ = opened()
    assert cfg.pixel_size == 3
    r = row(panel, "Pixel size")
    click(panel, r.parts[1][0].center)               # +
    assert cfg.pixel_size == PIXEL_STEPS[PIXEL_STEPS.index(3) + 1]
    click(panel, row(panel, "Pixel size").parts[0][0].center)   # -
    click(panel, row(panel, "Pixel size").parts[0][0].center)
    assert cfg.pixel_size == PIXEL_STEPS[0]
    click(panel, row(panel, "Pixel size").parts[0][0].center)   # clamps at the tightest
    assert cfg.pixel_size == PIXEL_STEPS[0]


def test_bit_depth_stepper_stays_in_range():
    cfg, panel, _ = opened()
    for _ in range(6):
        click(panel, row(panel, "Bit depth").parts[1][0].center)
    assert cfg.bit_depth == 4


def test_slider_click_sets_a_value_inside_its_range():
    cfg, panel, _ = opened()
    r = row(panel, "Wave strength")
    track = r.parts[0][0]
    click(panel, (track.x + track.width // 2, track.centery))
    assert 0.5 < cfg.wave_strength < 0.65
    click(panel, (track.right + 50, track.centery)) if panel._panel.collidepoint((track.right + 50, track.centery)) else None
    click(panel, (track.x, track.centery))
    assert cfg.wave_strength == 0.1


def test_slider_drag_follows_the_mouse():
    cfg, panel, _ = opened()
    track = row(panel, "Wave softness").parts[0][0]
    click(panel, (track.x, track.centery))
    assert cfg.wave_softness == 0.0
    panel.handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(track.right, track.centery), rel=(0, 0), buttons=(1, 0, 0)), W, H)
    assert cfg.wave_softness == 1.0
    panel.handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, pos=(track.right, track.centery), button=1), W, H)
    assert panel._drag is None


def test_character_options_appear_only_in_character_mode():
    cfg, panel, _ = opened()
    labels = [r.label for r in panel.layout(W, H)]
    assert "Your characters (paste or type; any font)" not in labels and "Pixel size" in labels
    r = row(panel, "Draw with")
    chars_pill = next(rect for rect, payload in r.parts if payload == "chars")
    click(panel, chars_pill.center)
    assert cfg.render_mode == "chars"
    labels = [r.label for r in panel.layout(W, H)]
    assert "Your characters (paste or type; any font)" in labels and "Character size" in labels
    assert "Pixel size" not in labels


def _chars_mode():
    cfg, panel, calls = opened()
    cfg.render_mode = "chars"
    return cfg, panel, calls


def test_typing_your_own_characters():
    cfg, panel, calls = _chars_mode()
    click(panel, row(panel, "Your characters (paste or type; any font)").parts[0][0].center)
    assert panel.focus == "chars"
    assert text(panel, "ab") and text(panel, "♥")
    assert cfg.glyph_chars == "ab♥" and "glyph_chars" in calls["changed"]
    key(panel, pygame.K_BACKSPACE)
    assert cfg.glyph_chars == "ab"
    key(panel, pygame.K_BACKSPACE, pygame.KMOD_CTRL)
    assert cfg.glyph_chars == ""
    assert key(panel, pygame.K_RETURN) and panel.focus is None


def test_hotkeys_are_swallowed_while_typing_but_not_otherwise():
    cfg, panel, _ = _chars_mode()
    assert not key(panel, pygame.K_b)                # panel open, not typing: main handles B
    click(panel, row(panel, "Your characters (paste or type; any font)").parts[0][0].center)
    assert key(panel, pygame.K_b)                    # typing: B must not trigger a hotkey


def test_control_characters_are_not_accepted_as_text():
    cfg, panel, _ = _chars_mode()
    click(panel, row(panel, "Your characters (paste or type; any font)").parts[0][0].center)
    text(panel, "x\r\n\t\x00y")
    assert cfg.glyph_chars == "xy"


def test_quick_sets_fill_the_character_box():
    cfg, panel, _ = _chars_mode()
    r = row(panel, "Quick sets")
    click(panel, r.parts[0][0].center)
    assert cfg.glyph_chars == "01"


def test_builtin_shapes_toggle_and_keep_a_stable_order():
    cfg, panel, _ = _chars_mode()
    assert "star" not in cfg.glyph_shapes
    r = row(panel, "Built-in shapes")
    star = next(rect for rect, (label, payload) in r.parts if payload == "star")
    click(panel, star.center)
    assert "star" in cfg.glyph_shapes
    assert cfg.glyph_shapes == [n for n in ["circle", "square", "triangle", "diamond", "plus", "cross", "star"] if n in cfg.glyph_shapes]
    click(panel, next(rect for rect, (label, payload) in row(panel, "Built-in shapes").parts if payload == "star").center)
    assert "star" not in cfg.glyph_shapes


def test_picking_a_font_and_filtering_the_list():
    cfg, panel, _ = _chars_mode()
    panel.layout(W, H)
    panel.scroll = max(0, panel._content_h - H)      # scroll down to the font list
    r = row(panel, "Font for your characters")
    click(panel, r.parts[0][0].center)               # focus the filter box
    assert panel.focus == "fontfilter"
    text(panel, "wingd")
    names = panel._filtered_fonts()
    assert names and all("wingd" in n for n in names)
    key(panel, pygame.K_RETURN)
    r = next(x for x in panel.layout(W, H) if x.kind == "fontlist")
    click(panel, r.parts[1][0].center)               # first match
    assert cfg.glyph_font == names[0]
    panel.font_filter = ""
    r = next(x for x in panel.layout(W, H) if x.kind == "fontlist")
    click(panel, r.parts[1][0].center)               # "(default)" is first
    assert cfg.glyph_font == ""


def test_reshuffle_button_calls_back():
    cfg, panel, calls = opened()
    click(panel, row(panel, "Reshuffle the layout now").rect.center)
    assert calls["reshuffled"] == 1


def test_clicks_outside_the_panel_pass_through():
    cfg, panel, _ = opened()
    assert not click(panel, (50, 50))
    assert click(panel, (W - 5, 5))                  # inside: consumed even on empty space


def test_closed_panel_consumes_nothing():
    cfg, panel, _ = make()
    assert not click(panel, (W - 5, 5))
    assert not text(panel, "x")


def test_character_size_steps():
    cfg, panel, _ = _chars_mode()
    r = row(panel, "Character size")
    click(panel, r.parts[1][0].center)
    assert cfg.glyph_cell == GLYPH_CELL_STEPS[GLYPH_CELL_STEPS.index(12) + 1]
