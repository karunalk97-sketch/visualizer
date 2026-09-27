import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pygame

from visualizer.renderer import BitmapRenderer, bar_height

pygame.init()
pygame.display.set_mode((800, 450))
FONT = pygame.font.SysFont("segoeui,arial,helvetica,sans", 14)


def test_short_text_is_left_alone_and_long_text_is_cut_with_an_ellipsis():
    assert BitmapRenderer.fit_text(FONT, "Short", 400) == "Short"
    cut = BitmapRenderer.fit_text(FONT, "A very long song title that cannot possibly fit " * 3, 200)
    assert cut.endswith("\u2026") and FONT.size(cut)[0] <= 200


def test_no_room_gives_no_text_instead_of_a_stray_ellipsis():
    assert BitmapRenderer.fit_text(FONT, "Anything at all", 3) == ""


def test_the_status_bar_never_lets_left_text_run_into_the_right_text():
    r = BitmapRenderer(100, 50, 800, 450)
    r.render_field(np.zeros((50, 100), np.float32), [(0, 0, 0), (255, 255, 255)], "L" * 400, "Spotify   Tab: settings   1-bit")
    bar = r.screen.subsurface(pygame.Rect(0, r.field_h, 800, r.bar_h))
    pixels = pygame.surfarray.array3d(bar).sum(axis=2).sum(axis=1)       # brightness per column
    right = r._font.size("Spotify   Tab: settings   1-bit")[0]              # the bar's own font, not the test font
    pad = max(10, r.bar_h // 2)
    start = 800 - right - pad                                             # where the right text begins
    gap_cols = pixels[start - 9: start - 1]                               # the strip just left of it (the bar leaves one pad of space)
    assert gap_cols.max() == 0                                            # nothing drawn there: the left text stopped short


def test_bar_height_is_slim_and_bounded():
    assert bar_height(100) == 24 and bar_height(2000) == 40 and 24 <= bar_height(720) <= 40
