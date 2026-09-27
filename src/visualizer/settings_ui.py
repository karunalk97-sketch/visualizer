"""In-window settings panel (Tab, or click the status bar). Drawn over the picture
in the app's own black-and-white style, so it looks the same on every platform.

Three tabs (Look / Layers / Characters) keep everything on one screen -- no
scrolling -- with large click targets, hover feedback and a Randomize button that
is always visible. Everything changes live and lasts for the session only. The
panel knows nothing about the visualizer: it reads and writes a Config and calls
back.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pygame

from .config import GLYPH_CELL_STEPS, GLYPH_SETS, PIXEL_STEPS, Config
from .glyphs import glyph_set

BG = (10, 10, 10, 250)
FG = (228, 228, 228)
DIM = (128, 128, 128)
LINE = (62, 62, 62)
HOVER = (28, 28, 28)
ACCENT = (245, 245, 245)
FONT_CANDIDATES = "segoeui,helveticaneue,helvetica,arial,dejavusans,sans"
TABS = [("look", "Look"), ("layers", "Layers"), ("chars", "Characters")]


@dataclass
class Row:
    kind: str               # toggle | stepper | slider | choice | chips | preview | button | note
    label: str = ""
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    parts: list = field(default_factory=list)   # (rect, payload) for the clickable pieces
    ctl: "Ctl | None" = None


@dataclass
class Ctl:
    kind: str
    label: str
    tab: str
    get: Callable = lambda: None
    set: Callable = lambda v: None
    options: list = field(default_factory=list)
    lo: float = 0.0
    hi: float = 1.0
    fmt: str = "{:.2f}"
    action: Callable | None = None
    visible: Callable = lambda: True


class SettingsPanel:
    def __init__(self, cfg: Config, on_change: Callable[[str], None], on_reshuffle: Callable[[], None],
                 on_randomize: Callable[[], None]) -> None:
        self.cfg = cfg
        self.on_change = on_change
        self.on_reshuffle = on_reshuffle
        self.on_randomize = on_randomize
        self.visible = False
        self.tab = "look"
        self.hover: Row | None = None
        self._drag: Row | None = None
        self._rows: list[Row] = []
        self._header: dict[str, pygame.Rect] = {}
        self._panel = pygame.Rect(0, 0, 0, 0)
        self._fonts: dict[int, pygame.font.Font] = {}
        self._text: dict[tuple, pygame.Surface] = {}
        self._bg: tuple[tuple, pygame.Surface] | None = None
        self._preview: tuple[tuple, list[pygame.Surface]] | None = None
        self._cursor_hand = False
        self._controls = self._build_controls()

    # -- controls ----------------------------------------------------------------

    def _s(self, name: str) -> Callable:
        def setter(v):
            setattr(self.cfg, name, v)
            self.on_change(name)
        return setter

    def _build_controls(self) -> list[Ctl]:
        c, s = self.cfg, self._s
        chars_on = lambda: c.render_mode == "chars"    # noqa: E731
        pixels_on = lambda: c.render_mode == "pixels"  # noqa: E731

        def toggle_set(key):
            cur = list(c.glyph_sets)
            if key in cur:
                if len(cur) > 1:            # always keep at least one set
                    cur.remove(key)
            else:
                cur.append(key)
            c.glyph_sets = [k for k, _ in GLYPH_SETS if k in cur]
            self.on_change("glyph_sets")

        return [
            Ctl("choice", "Draw with", "look", lambda: c.render_mode, s("render_mode"), options=[("pixels", "Pixels"), ("chars", "Characters")]),
            Ctl("stepper", "Pixel size", "look", lambda: c.pixel_size, s("pixel_size"), options=PIXEL_STEPS, fmt="{} px", visible=pixels_on),
            Ctl("stepper", "Character size", "look", lambda: c.glyph_cell, s("glyph_cell"), options=GLYPH_CELL_STEPS, fmt="{} px", visible=chars_on),
            Ctl("stepper", "Bit depth", "look", lambda: c.bit_depth, s("bit_depth"), options=[1, 2, 3, 4], fmt="{}-bit"),
            Ctl("slider", "3D depth", "look", lambda: c.depth, s("depth"), lo=0.0, hi=1.0),
            Ctl("slider", "Sensitivity", "look", lambda: c.gain, s("gain"), lo=0.4, hi=3.0, fmt="{:.1f}x"),
            Ctl("toggle", "Fullscreen", "look", lambda: c.fullscreen, s("fullscreen")),
            Ctl("toggle", "New layout on every song", "look", lambda: c.reshuffle_on_new_song, s("reshuffle_on_new_song")),
            Ctl("toggle", "Track and status bar text", "look", lambda: c.show_now_playing, s("show_now_playing")),
            Ctl("button", "Reshuffle the layout now", "look", action=self.on_reshuffle),

            Ctl("toggle", "Shapes", "layers", lambda: c.show_shapes, s("show_shapes")),
            Ctl("slider", "Overlap inversion", "layers", lambda: c.overlap_invert, s("overlap_invert"), lo=0.0, hi=1.0),
            Ctl("toggle", "Foam wave", "layers", lambda: c.waves, s("waves")),
            Ctl("slider", "Wave strength", "layers", lambda: c.wave_strength, s("wave_strength"), lo=0.1, hi=1.0),
            Ctl("slider", "Wave softness", "layers", lambda: c.wave_softness, s("wave_softness"), lo=0.0, hi=1.0),
            Ctl("slider", "How often waves come", "layers", lambda: c.wave_rate, s("wave_rate"), lo=0.25, hi=3.0, fmt="{:.1f}x"),

            Ctl("note", "Switch “Draw with” to Characters on the Look tab to use these.", "chars", visible=pixels_on),
            Ctl("chips", "Character sets", "chars", lambda: c.glyph_sets, toggle_set, options=GLYPH_SETS),
            Ctl("preview", "Preview", "chars"),
            Ctl("choice", "Pick glyphs", "chars", lambda: c.glyph_mapping, s("glyph_mapping"),
                options=[("random", "Random"), ("brightness", "By brightness")]),
            Ctl("stepper", "Character size", "chars", lambda: c.glyph_cell, s("glyph_cell"), options=GLYPH_CELL_STEPS, fmt="{} px"),
        ]

    # -- open / close ------------------------------------------------------------

    def toggle(self) -> None:
        self.visible = not self.visible
        self._drag = None
        if not self.visible:
            self._set_cursor(False)

    def close(self) -> None:
        if self.visible:
            self.toggle()

    def set_tab(self, key: str) -> None:
        if key in dict(TABS):
            self.tab = key
            self._drag = None

    # -- layout ------------------------------------------------------------------

    def _font(self, px: int) -> pygame.font.Font:
        if px not in self._fonts:
            self._fonts[px] = pygame.font.SysFont(FONT_CANDIDATES, px)
        return self._fonts[px]

    def _text_surf(self, text: str, px: int, color) -> pygame.Surface:
        key = (text, px, color)
        surf = self._text.get(key)
        if surf is None:
            if len(self._text) > 600:
                self._text.clear()
            surf = self._font(px).render(text, True, color)
            self._text[key] = surf
        return surf

    def layout(self, screen_w: int, field_h: int) -> list[Row]:
        ui = max(0.75, min(1.7, field_h / 720))
        width = int(min(screen_w - 16, 366 * ui))
        self._panel = pygame.Rect(screen_w - width, 0, width, field_h)
        pad = int(18 * ui)
        x0, x1 = self._panel.x + pad, self._panel.right - pad
        f = self._font(int(15 * ui))

        self._header = {"close": pygame.Rect(x1 - int(32 * ui), pad - int(4 * ui), int(32 * ui), int(32 * ui))}
        ty = pad + int(40 * ui)
        th = int(34 * ui)
        gap = int(6 * ui)
        tw = (x1 - x0 - gap * (len(TABS) - 1)) // len(TABS)
        for i, (key, _) in enumerate(TABS):
            self._header["tab:" + key] = pygame.Rect(x0 + i * (tw + gap), ty, tw, th)
        foot_h = int(44 * ui)
        self._header["randomize"] = pygame.Rect(x0, field_h - pad - foot_h - int(20 * ui), x1 - x0, foot_h)

        body_top = ty + th + int(14 * ui)
        body_bottom = self._header["randomize"].top - int(12 * ui)
        controls = [c for c in self._controls if c.tab == self.tab and c.visible()]

        def heights(row_h: int) -> list[int]:
            out = []
            for c in controls:
                if c.kind == "slider":
                    out.append(int(row_h * 1.4))
                elif c.kind == "chips":
                    out.append(int(row_h * 0.75) + int(row_h * 0.85) * (1 + (len(c.options) > 3 and width < 380 * ui)))
                elif c.kind == "preview":
                    out.append(int(row_h * 1.15))
                elif c.kind == "note":
                    out.append(int(row_h * 1.4))
                else:
                    out.append(row_h)
            return out

        row_h = int(38 * ui)
        while sum(heights(row_h)) > body_bottom - body_top and row_h > int(30 * ui):
            row_h -= 1                                    # short window: tighten rows, still easy to click

        rows: list[Row] = []
        y = body_top
        for c, h in zip(controls, heights(row_h)):
            r = Row(c.kind, c.label, pygame.Rect(x0, y, x1 - x0, h), [], c)
            k = c.kind
            if k == "stepper":
                bw = int(36 * ui)
                val_w = int(90 * ui)
                r.parts = [(pygame.Rect(x1 - bw * 2 - val_w, y + (h - bw) // 2, bw, bw), -1),
                           (pygame.Rect(x1 - bw, y + (h - bw) // 2, bw, bw), +1)]
            elif k == "slider":
                r.parts = [(pygame.Rect(x0, y + int(h * 0.62) - int(12 * ui), x1 - x0, int(24 * ui)), None)]
            elif k == "choice":
                pill_w = int((x1 - x0) * 0.6) // len(c.options)
                pw = pill_w * len(c.options)
                r.parts = [(pygame.Rect(x1 - pw + i * pill_w + 1, y + (h - int(row_h * 0.8)) // 2, pill_w - 2, int(row_h * 0.8)), o[0])
                           for i, o in enumerate(c.options)]
            elif k == "chips":
                cx, cy, ch_h = x0, y + int(row_h * 0.75), int(row_h * 0.75)
                for key, label in c.options:
                    w = f.size(label)[0] + int(26 * ui)
                    if cx + w > x1:
                        cx, cy = x0, cy + ch_h + int(6 * ui)
                    r.parts.append((pygame.Rect(cx, cy, w, ch_h), key))
                    cx += w + int(8 * ui)
            elif k == "button":
                r.parts = [(pygame.Rect(x0, y + int(3 * ui), x1 - x0, h - int(6 * ui)), None)]
            rows.append(r)
            y += h
        self._rows = rows
        return rows

    # -- hit testing -------------------------------------------------------------

    def _row_at(self, pos) -> Row | None:
        for r in self._rows:
            if r.rect.collidepoint(pos) and r.kind not in ("note", "preview"):
                return r
        return None

    def interactive_at(self, pos) -> bool:
        if not self.visible or not self._panel.collidepoint(pos):
            return False
        if any(rect.collidepoint(pos) for rect in self._header.values()):
            return True
        return self._row_at(pos) is not None

    def _set_cursor(self, hand: bool) -> None:
        if hand == self._cursor_hand:
            return
        self._cursor_hand = hand
        try:
            pygame.mouse.set_cursor(pygame.SYSTEM_CURSOR_HAND if hand else pygame.SYSTEM_CURSOR_ARROW)
        except Exception:
            pass

    # -- events ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, screen_w: int, field_h: int) -> bool:
        """Returns True if the panel consumed the event."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self.toggle()
            return True
        if not self.visible:
            return False
        self.layout(screen_w, field_h)

        if event.type == pygame.KEYDOWN:
            keys = [k for k, _ in TABS]
            if event.key == pygame.K_ESCAPE:
                self.close()
                return True
            if event.key in (pygame.K_LEFT, pygame.K_RIGHT):
                step = -1 if event.key == pygame.K_LEFT else 1
                self.set_tab(keys[(keys.index(self.tab) + step) % len(keys)])
                return True
            return False
        if event.type == pygame.MOUSEMOTION:
            self.hover = self._row_at(event.pos) if self._panel.collidepoint(event.pos) else None
            if self._drag is not None:
                self._set_slider(self._drag, event.pos[0])
                return True
            self._set_cursor(self.interactive_at(event.pos))
            return self._panel.collidepoint(event.pos)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self._panel.collidepoint(event.pos):
                return False
            self._click(event.pos)
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self._drag is not None:
                self._drag = None
                return True
            return self._panel.collidepoint(event.pos)
        if event.type == pygame.MOUSEWHEEL:
            return self._panel.collidepoint(pygame.mouse.get_pos())
        return False

    def _click(self, pos) -> None:
        if self._header["close"].collidepoint(pos):
            self.close()
            return
        for key, _ in TABS:
            if self._header["tab:" + key].collidepoint(pos):
                self.set_tab(key)
                return
        if self._header["randomize"].collidepoint(pos):
            self.on_randomize()
            return
        r = self._row_at(pos)
        if r is None:
            return
        ctl, k = r.ctl, r.kind
        if k == "toggle":
            ctl.set(not ctl.get())
        elif k == "button" and ctl.action:
            ctl.action()
        elif k == "slider":
            self._drag = r
            self._set_slider(r, pos[0])
        elif k == "stepper":
            for rect, d in r.parts:
                if rect.collidepoint(pos):
                    opts, cur = ctl.options, ctl.get()
                    i = min(range(len(opts)), key=lambda j: abs(opts[j] - cur))
                    ctl.set(opts[max(0, min(len(opts) - 1, i + d))])
        elif k in ("choice", "chips"):
            for rect, payload in r.parts:
                if rect.collidepoint(pos):
                    ctl.set(payload)

    def _set_slider(self, r: Row, x: int) -> None:
        track = r.parts[0][0]
        t = max(0.0, min(1.0, (x - track.x) / max(1, track.width)))
        ctl = r.ctl
        ctl.set(round(ctl.lo + t * (ctl.hi - ctl.lo), 3))

    # -- drawing -----------------------------------------------------------------

    def _preview_tiles(self, ui: float) -> list[pygame.Surface]:
        cell = int(28 * ui)
        sig = (tuple(self.cfg.glyph_sets), cell)
        if self._preview is None or self._preview[0] != sig:
            tiles = []
            seen = set()
            for cov in glyph_set(list(self.cfg.glyph_sets), 24):
                key = cov.tobytes()
                if key in seen:
                    continue
                seen.add(key)
                arr = (np.clip(cov, 0, 1) * 255).astype(np.uint8)
                surf = pygame.surfarray.make_surface(np.repeat(arr.T[:, :, None], 3, axis=2))
                tiles.append(pygame.transform.smoothscale(surf, (cell, cell)))
            self._preview = (sig, tiles)
        return self._preview[1]

    def draw(self, surface: pygame.Surface, screen_w: int, field_h: int) -> None:
        if not self.visible:
            return
        rows = self.layout(screen_w, field_h)
        ui = max(0.75, min(1.7, field_h / 720))
        f, fs, fb = self._font(int(15 * ui)), self._font(int(12 * ui)), self._font(int(19 * ui))

        bg_key = (self._panel.size,)
        if self._bg is None or self._bg[0] != bg_key:
            bg = pygame.Surface(self._panel.size, pygame.SRCALPHA)
            bg.fill(BG)
            pygame.draw.line(bg, LINE, (0, 0), (0, self._panel.height))
            self._bg = (bg_key, bg)
        surface.blit(self._bg[1], self._panel.topleft)
        clip = surface.get_clip()
        surface.set_clip(self._panel)

        def text(s, px, color, pos, right=False, center=False, midleft=False):
            img = self._text_surf(s, px, color)
            rect = img.get_rect()
            if right:
                rect.midright = pos
            elif center:
                rect.center = pos
            elif midleft:
                rect.midleft = pos
            else:
                rect.topleft = pos
            surface.blit(img, rect)

        def pill(rect, label, on, hover=False):
            radius = rect.height // 2
            if on:
                pygame.draw.rect(surface, ACCENT, rect, border_radius=radius)
            else:
                pygame.draw.rect(surface, HOVER if hover else BG[:3], rect, border_radius=radius)
                pygame.draw.rect(surface, DIM if hover else LINE, rect, 1, border_radius=radius)
            text(label, int(14 * ui), (0, 0, 0) if on else FG, rect.center, center=True)

        mouse = pygame.mouse.get_pos()
        hp = self._header
        text("Settings", int(19 * ui), ACCENT, (self._panel.x + int(18 * ui), hp["close"].centery), midleft=True)
        hc = hp["close"].collidepoint(mouse)
        pygame.draw.rect(surface, HOVER if hc else BG[:3], hp["close"], border_radius=8)
        cx, cy, d = hp["close"].centerx, hp["close"].centery, int(6 * ui)
        pygame.draw.line(surface, ACCENT if hc else FG, (cx - d, cy - d), (cx + d, cy + d), 2)
        pygame.draw.line(surface, ACCENT if hc else FG, (cx - d, cy + d), (cx + d, cy - d), 2)
        for key, label in TABS:
            rect = hp["tab:" + key]
            pill(rect, label, self.tab == key, rect.collidepoint(mouse))

        for r in rows:
            ctl, k, rc = r.ctl, r.kind, r.rect
            if k in ("toggle", "stepper", "slider", "choice", "button") and (r is self.hover or r is self._drag) and k != "button":
                pygame.draw.rect(surface, HOVER, rc.inflate(int(10 * ui), 0), border_radius=8)
            if k == "toggle":
                text(ctl.label, int(15 * ui), FG, (rc.x, rc.centery), midleft=True)
                sw = pygame.Rect(0, 0, int(42 * ui), int(22 * ui))
                sw.midright = (rc.right, rc.centery)
                on = bool(ctl.get())
                pygame.draw.rect(surface, ACCENT if on else BG[:3], sw, border_radius=sw.height // 2)
                pygame.draw.rect(surface, ACCENT if on else DIM, sw, 1, border_radius=sw.height // 2)
                knob = sw.height - 8
                pygame.draw.circle(surface, (0, 0, 0) if on else DIM,
                                   (sw.right - knob // 2 - 5 if on else sw.left + knob // 2 + 5, sw.centery), knob // 2)
            elif k == "stepper":
                text(ctl.label, int(15 * ui), FG, (rc.x, rc.centery), midleft=True)
                (minus, _), (plus, _) = r.parts
                for rect, sign in ((minus, "−"), (plus, "+")):
                    hov = rect.collidepoint(mouse)
                    pygame.draw.rect(surface, HOVER if hov else BG[:3], rect, border_radius=8)
                    pygame.draw.rect(surface, DIM if hov else LINE, rect, 1, border_radius=8)
                    text(sign, int(18 * ui), FG, rect.center, center=True)
                text(ctl.fmt.format(ctl.get()), int(15 * ui), ACCENT, ((minus.right + plus.left) // 2, rc.centery), center=True)
            elif k == "slider":
                val = ctl.get()
                text(ctl.label, int(15 * ui), FG, (rc.x, rc.y + int(rc.height * 0.16)), midleft=True)
                text(ctl.fmt.format(val), int(15 * ui), ACCENT, (rc.right, rc.y + int(rc.height * 0.16)), right=True)
                track = r.parts[0][0]
                mid = track.centery
                t = (val - ctl.lo) / (ctl.hi - ctl.lo)
                kx = track.x + int(t * track.width)
                pygame.draw.line(surface, LINE, (track.x, mid), (track.right, mid), 4)
                pygame.draw.line(surface, ACCENT, (track.x, mid), (kx, mid), 4)
                big = r is self._drag or r is self.hover
                pygame.draw.circle(surface, ACCENT, (kx, mid), int((9 if big else 7) * ui))
            elif k == "choice":
                text(ctl.label, int(15 * ui), FG, (rc.x, rc.centery), midleft=True)
                for rect, payload in r.parts:
                    label = next(lbl for key, lbl in ctl.options if key == payload)
                    pill(rect, label, ctl.get() == payload, rect.collidepoint(mouse))
            elif k == "chips":
                text(ctl.label, int(15 * ui), FG, (rc.x, rc.y + int(rc.height * 0.14)), midleft=True)
                chosen = ctl.get()
                for rect, key in r.parts:
                    label = next(lbl for kk, lbl in ctl.options if kk == key)
                    pill(rect, label, key in chosen, rect.collidepoint(mouse))
            elif k == "preview":
                x = rc.x
                for tile in self._preview_tiles(ui):
                    if x + tile.get_width() > rc.right:
                        break
                    surface.blit(tile, (x, rc.y + (rc.height - tile.get_height()) // 2))
                    x += tile.get_width() + int(4 * ui)
            elif k == "button":
                rect = r.parts[0][0]
                hov = rect.collidepoint(mouse)
                pygame.draw.rect(surface, HOVER if hov else BG[:3], rect, border_radius=8)
                pygame.draw.rect(surface, DIM if hov else LINE, rect, 1, border_radius=8)
                text(ctl.label, int(15 * ui), FG, rect.center, center=True)
            elif k == "note":
                words, line, y = ctl.label.split(), "", rc.y
                for w in words:
                    trial = (line + " " + w).strip()
                    if self._font(int(12 * ui)).size(trial)[0] > rc.width and line:
                        text(line, int(12 * ui), DIM, (rc.x, y))
                        y += int(16 * ui)
                        line = w
                    else:
                        line = trial
                text(line, int(12 * ui), DIM, (rc.x, y))

        rz = hp["randomize"]
        hov = rz.collidepoint(mouse)
        pygame.draw.rect(surface, ACCENT if hov else FG, rz, border_radius=10)
        text("Randomize", int(17 * ui), (0, 0, 0), rz.center, center=True)
        text("Tab or Esc to close  ·  Space also randomizes  ·  ← → switch tabs", int(11 * ui), DIM,
             (self._panel.centerx, rz.bottom + int(14 * ui)), center=True)
        surface.set_clip(clip)
