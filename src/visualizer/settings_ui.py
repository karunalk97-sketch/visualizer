"""In-window settings panel (Tab). Drawn over the picture in the app's own
black-and-white style, so it looks the same on every platform.

Everything changes live and is saved when the panel closes. The panel knows
nothing about the visualizer: it reads and writes a Config and calls back.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable

import pygame

from .config import Config, GLYPH_CELL_STEPS, PIXEL_STEPS
from .glyphs import BUILTIN_SHAPES

BG = (10, 10, 10, 248)
FG = (225, 225, 225)
DIM = (128, 128, 128)
LINE = (58, 58, 58)
ACCENT = (240, 240, 240)
FONT_CANDIDATES = "segoeui,helveticaneue,helvetica,arial,dejavusans,sans"
CHAR_PRESETS = [("01", "01"), ("+×=−", "+×=−"), ("░▒▓█", "░▒▓█"),
                ("▲●■◆", "▲●■◆"), ("·•●○", "·•●○")]
FONT_ROWS = 6


def clipboard_text() -> str:
    """Best-effort paste; '' when unavailable."""
    if sys.platform == "win32":
        try:
            import ctypes

            u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
            u32.GetClipboardData.restype = ctypes.c_void_p
            k32.GlobalLock.restype = ctypes.c_void_p
            k32.GlobalLock.argtypes = [ctypes.c_void_p]
            k32.GlobalUnlock.argtypes = [ctypes.c_void_p]
            if not u32.OpenClipboard(None):
                return ""
            try:
                handle = u32.GetClipboardData(13)  # CF_UNICODETEXT
                if not handle:
                    return ""
                ptr = k32.GlobalLock(handle)
                try:
                    return ctypes.wstring_at(ptr)
                finally:
                    k32.GlobalUnlock(handle)
            finally:
                u32.CloseClipboard()
        except Exception:
            return ""
    try:
        pygame.scrap.init()
        raw = pygame.scrap.get("text/plain;charset=utf-8") or pygame.scrap.get(pygame.SCRAP_TEXT)
        return raw.decode("utf-8", "ignore").rstrip("\x00") if raw else ""
    except Exception:
        return ""


@dataclass
class Row:
    kind: str               # section | toggle | stepper | slider | choice | chips | text | fontlist | button | note
    label: str = ""
    rect: pygame.Rect = field(default_factory=lambda: pygame.Rect(0, 0, 0, 0))
    parts: list = field(default_factory=list)   # (rect, payload) for the clickable pieces
    ctl: "Ctl | None" = None


@dataclass
class Ctl:
    kind: str
    label: str
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
                 on_close: Callable[[], None] | None = None) -> None:
        self.cfg = cfg
        self.on_change = on_change
        self.on_reshuffle = on_reshuffle
        self.on_close = on_close
        self.visible = False
        self.focus: str | None = None          # "chars" | "fontfilter" while typing
        self.font_filter = ""
        self.font_scroll = 0
        self.scroll = 0
        self._drag: Row | None = None
        self._rows: list[Row] = []
        self._panel = pygame.Rect(0, 0, 0, 0)
        self._content_h = 0
        self._fonts: dict[int, pygame.font.Font] = {}
        self._all_fonts = ["(default)"] + sorted(n for n in pygame.font.get_fonts() if n)
        self._controls = self._build_controls()

    # -- controls ----------------------------------------------------------------

    def _s(self, name: str) -> Callable:
        def setter(v):
            setattr(self.cfg, name, v)
            self.on_change(name)
        return setter

    def _build_controls(self) -> list[Ctl]:
        c, s = self.cfg, self._s
        chars_on = lambda: c.render_mode == "chars"
        pixels_on = lambda: c.render_mode == "pixels"

        def toggle_shape(name):
            cur = list(c.glyph_shapes)
            (cur.remove if name in cur else cur.append)(name)
            c.glyph_shapes = [n for n in BUILTIN_SHAPES if n in cur]
            self.on_change("glyph_shapes")

        return [
            Ctl("section", "DISPLAY"),
            Ctl("toggle", "Fullscreen", lambda: c.fullscreen, s("fullscreen")),
            Ctl("stepper", "Bit depth", lambda: c.bit_depth, s("bit_depth"), options=[1, 2, 3, 4], fmt="{}-bit"),
            Ctl("choice", "Draw with", lambda: c.render_mode, s("render_mode"), options=[("pixels", "Pixels"), ("chars", "Characters")]),
            Ctl("stepper", "Pixel size", lambda: c.pixel_size, s("pixel_size"), options=PIXEL_STEPS, fmt="{} px", visible=pixels_on),
            Ctl("stepper", "Character size", lambda: c.glyph_cell, s("glyph_cell"), options=GLYPH_CELL_STEPS, fmt="{} px", visible=chars_on),
            Ctl("section", "CHARACTERS", visible=chars_on),
            Ctl("choice", "Pick each cell's character", lambda: c.glyph_mapping, s("glyph_mapping"),
                options=[("random", "Random"), ("brightness", "By brightness")], visible=chars_on),
            Ctl("chips", "Built-in shapes", lambda: c.glyph_shapes, toggle_shape, options=BUILTIN_SHAPES, visible=chars_on),
            Ctl("text", "Your characters (paste or type; any font)", lambda: c.glyph_chars, s("glyph_chars"), visible=chars_on),
            Ctl("presets", "Quick sets", lambda: c.glyph_chars, s("glyph_chars"), options=CHAR_PRESETS, visible=chars_on),
            Ctl("fontlist", "Font for your characters", lambda: c.glyph_font, s("glyph_font"), visible=chars_on),
            Ctl("section", "LAYERS"),
            Ctl("toggle", "Shapes", lambda: c.show_shapes, s("show_shapes")),
            Ctl("slider", "Overlap inversion", lambda: c.overlap_invert, s("overlap_invert"), lo=0.0, hi=1.0),
            Ctl("toggle", "Waves", lambda: c.waves, s("waves")),
            Ctl("slider", "Wave strength", lambda: c.wave_strength, s("wave_strength"), lo=0.1, hi=1.0),
            Ctl("slider", "Wave softness", lambda: c.wave_softness, s("wave_softness"), lo=0.0, hi=1.0),
            Ctl("slider", "How often waves appear", lambda: c.wave_rate, s("wave_rate"), lo=0.25, hi=3.0, fmt="{:.1f}x"),
            Ctl("button", "Reshuffle the layout now", action=self.on_reshuffle),
            Ctl("section", "BEHAVIOUR"),
            Ctl("toggle", "New layout on every new song", lambda: c.reshuffle_on_new_song, s("reshuffle_on_new_song")),
            Ctl("toggle", "Show track and status bar text", lambda: c.show_now_playing, s("show_now_playing")),
            Ctl("slider", "Sensitivity", lambda: c.gain, s("gain"), lo=0.4, hi=3.0, fmt="{:.1f}x"),
            Ctl("note", "Tab or Esc closes. Changes are live and saved when you close."),
        ]

    # -- open / close ------------------------------------------------------------

    def toggle(self) -> None:
        if self.visible:
            self.close()
        else:
            self.visible = True

    def close(self) -> None:
        if not self.visible:
            return
        self.visible = False
        self.focus = None
        self._drag = None
        if self.on_close:
            self.on_close()

    # -- layout ------------------------------------------------------------------

    def _font(self, px: int) -> pygame.font.Font:
        if px not in self._fonts:
            self._fonts[px] = pygame.font.SysFont(FONT_CANDIDATES, px)
        return self._fonts[px]

    def layout(self, screen_w: int, field_h: int) -> list[Row]:
        ui = max(1.0, min(1.8, field_h / 720))
        width = int(min(screen_w - 20, 430 * ui))
        self._panel = pygame.Rect(screen_w - width, 0, width, field_h)
        pad = int(16 * ui)
        row_h = int(30 * ui)
        f = self._font(int(15 * ui))
        x0, x1 = self._panel.x + pad, self._panel.right - pad
        y = pad - self.scroll
        rows: list[Row] = []

        def add(kind, ctl, h, parts=None):
            nonlocal y
            r = Row(kind, ctl.label, pygame.Rect(x0, y, x1 - x0, h), parts or [], ctl)
            rows.append(r)
            y += h
            return r

        for ctl in self._controls:
            if not ctl.visible():
                continue
            k = ctl.kind
            if k == "section":
                y += int(8 * ui)
                add(k, ctl, int(24 * ui))
            elif k in ("toggle", "button", "note"):
                add(k, ctl, row_h if k != "note" else int(40 * ui))
            elif k == "stepper":
                r = add(k, ctl, row_h)
                bw = int(26 * ui)
                val_w = int(86 * ui)
                minus = pygame.Rect(x1 - bw * 2 - val_w, r.rect.y + 2, bw, row_h - 4)
                plus = pygame.Rect(x1 - bw, r.rect.y + 2, bw, row_h - 4)
                r.parts = [(minus, -1), (plus, +1)]
            elif k == "slider":
                r = add(k, ctl, int(row_h * 1.35))
                r.parts = [(pygame.Rect(x0, r.rect.y + int(row_h * 0.85), x1 - x0, int(12 * ui)), None)]
            elif k == "choice":
                r = add(k, ctl, int(row_h * 1.65))
                gap, total = int(6 * ui), len(ctl.options)
                w = (x1 - x0 - gap * (total - 1)) // total
                r.parts = [(pygame.Rect(x0 + i * (w + gap), r.rect.y + int(row_h * 0.85), w, int(row_h * 0.7)), o[0])
                           for i, o in enumerate(ctl.options)]
            elif k in ("chips", "presets"):
                r = Row(k, ctl.label, pygame.Rect(x0, y, x1 - x0, 0), [], ctl)
                cx, cy, ch_h = x0, y + int(row_h * 0.85), int(row_h * 0.72)
                for opt in ctl.options:
                    label, payload = (opt, opt) if isinstance(opt, str) else opt
                    w = f.size(label)[0] + int(20 * ui)
                    if cx + w > x1:
                        cx, cy = x0, cy + ch_h + int(5 * ui)
                    r.parts.append((pygame.Rect(cx, cy, w, ch_h), (label, payload)))
                    cx += w + int(6 * ui)
                r.rect.height = cy + ch_h + int(8 * ui) - y
                rows.append(r)
                y += r.rect.height
            elif k == "text":
                r = add(k, ctl, int(row_h * 2.5))
                r.parts = [(pygame.Rect(x0, r.rect.y + int(row_h * 0.7), x1 - x0, int(row_h * 0.85)), "input")]
            elif k == "fontlist":
                filt = int(row_h * 0.85)
                r = add(k, ctl, int(row_h * 0.7) + filt + int(4 * ui) + FONT_ROWS * int(row_h * 0.8) + int(8 * ui))
                top = r.rect.y + int(row_h * 0.7)
                r.parts = [(pygame.Rect(x0, top, x1 - x0, filt), "filter")]
                names = self._filtered_fonts()
                self.font_scroll = max(0, min(self.font_scroll, max(0, len(names) - FONT_ROWS)))
                lh = int(row_h * 0.8)
                for i in range(FONT_ROWS):
                    if self.font_scroll + i < len(names):
                        r.parts.append((pygame.Rect(x0, top + filt + int(4 * ui) + i * lh, x1 - x0, lh), names[self.font_scroll + i]))
        self._content_h = y + self.scroll + pad
        self._rows = rows
        return rows

    def _filtered_fonts(self) -> list[str]:
        q = self.font_filter.lower().strip()
        return [n for n in self._all_fonts if q in n.lower()] if q else list(self._all_fonts)

    # -- events ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event, screen_w: int, field_h: int) -> bool:
        """Returns True if the panel consumed the event."""
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            self.toggle()
            return True
        if not self.visible:
            return False
        rows = self.layout(screen_w, field_h)

        if event.type == pygame.KEYDOWN:
            if self.focus:
                self._key_in_text(event)
                return True
            if event.key == pygame.K_ESCAPE:
                self.close()
                return True
            return False
        if event.type == pygame.TEXTINPUT and self.focus:
            self._type(event.text)
            return True
        if event.type == pygame.MOUSEWHEEL:
            mx, my = pygame.mouse.get_pos()
            if self._panel.collidepoint(mx, my):
                for r in rows:
                    if r.kind == "fontlist" and r.rect.collidepoint(mx, my):
                        self.font_scroll = max(0, self.font_scroll - event.y)
                        return True
                self.scroll = int(max(0, min(max(0, self._content_h - field_h), self.scroll - event.y * 40)))
                return True
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self._panel.collidepoint(event.pos):
                self.focus = None
                return False
            self.focus = None
            for r in rows:
                if r.rect.collidepoint(event.pos):
                    self._click(r, event.pos)
                    break
            return True
        if event.type == pygame.MOUSEMOTION and self._drag is not None:
            self._set_slider(self._drag, event.pos[0])
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._drag is not None:
            self._drag = None
            return True
        return self._panel.collidepoint(getattr(event, "pos", (-1, -1))) if event.type in (
            pygame.MOUSEBUTTONUP, pygame.MOUSEMOTION) else False

    def _click(self, r: Row, pos) -> None:
        ctl, k = r.ctl, r.kind
        if k == "toggle":
            ctl.set(not ctl.get())
        elif k == "button" and ctl.action:
            ctl.action()
        elif k == "stepper":
            for rect, d in r.parts:
                if rect.collidepoint(pos):
                    opts = ctl.options
                    cur = ctl.get()
                    i = min(range(len(opts)), key=lambda j: abs(opts[j] - cur))
                    ctl.set(opts[max(0, min(len(opts) - 1, i + d))])
        elif k == "slider":
            self._drag = r
            self._set_slider(r, pos[0])
        elif k == "choice":
            for rect, payload in r.parts:
                if rect.collidepoint(pos):
                    ctl.set(payload)
        elif k == "chips":
            for rect, (label, payload) in r.parts:
                if rect.collidepoint(pos):
                    ctl.set(payload)
        elif k == "presets":
            for rect, (label, payload) in r.parts:
                if rect.collidepoint(pos):
                    ctl.set(payload)
        elif k == "text":
            self.focus = "chars"
            pygame.key.start_text_input()
        elif k == "fontlist":
            for rect, payload in r.parts:
                if rect.collidepoint(pos):
                    if payload == "filter":
                        self.focus = "fontfilter"
                        pygame.key.start_text_input()
                    else:
                        ctl.set("" if payload == "(default)" else payload)

    def _set_slider(self, r: Row, x: int) -> None:
        track = r.parts[0][0]
        t = max(0.0, min(1.0, (x - track.x) / max(1, track.width)))
        ctl = r.ctl
        ctl.set(round(ctl.lo + t * (ctl.hi - ctl.lo), 3))

    def _type(self, text: str) -> None:
        text = "".join(ch for ch in text if ch.isprintable() or ch == " ")
        if not text:
            return
        if self.focus == "chars":
            self.cfg.glyph_chars = (self.cfg.glyph_chars + text)[:200]
            self.on_change("glyph_chars")
        elif self.focus == "fontfilter":
            self.font_filter = (self.font_filter + text)[:40]
            self.font_scroll = 0

    def _key_in_text(self, event: pygame.event.Event) -> None:
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_ESCAPE, pygame.K_TAB):
            self.focus = None
        elif event.key == pygame.K_BACKSPACE:
            if self.focus == "chars":
                self.cfg.glyph_chars = "" if ctrl else self.cfg.glyph_chars[:-1]
                self.on_change("glyph_chars")
            else:
                self.font_filter = "" if ctrl else self.font_filter[:-1]
                self.font_scroll = 0
        elif event.key == pygame.K_v and ctrl:
            self._type(clipboard_text())

    # -- drawing -----------------------------------------------------------------

    def draw(self, surface: pygame.Surface, screen_w: int, field_h: int) -> None:
        if not self.visible:
            return
        rows = self.layout(screen_w, field_h)
        ui = max(1.0, min(1.8, field_h / 720))
        f, fs = self._font(int(15 * ui)), self._font(int(12 * ui))
        panel = pygame.Surface(self._panel.size, pygame.SRCALPHA)
        panel.fill(BG)
        pygame.draw.line(panel, LINE, (0, 0), (0, self._panel.height))
        surface.blit(panel, self._panel.topleft)
        clip = surface.get_clip()
        surface.set_clip(self._panel)

        def text(s, font, color, pos, right=False, center=False):
            img = font.render(s, True, color)
            rect = img.get_rect()
            if right:
                rect.topright = pos
            elif center:
                rect.center = pos
            else:
                rect.topleft = pos
            surface.blit(img, rect)

        def pill(rect, label, on):
            pygame.draw.rect(surface, ACCENT if on else BG[:3], rect, border_radius=int(rect.height / 2))
            pygame.draw.rect(surface, ACCENT if on else LINE, rect, 1, border_radius=int(rect.height / 2))
            text(label, fs, (0, 0, 0) if on else FG, rect.center, center=True)

        for r in rows:
            ctl, k, rc = r.ctl, r.kind, r.rect
            if rc.bottom < 0 or rc.top > field_h:
                continue
            if k == "section":
                text(ctl.label, fs, DIM, (rc.x, rc.y + 6))
                pygame.draw.line(surface, LINE, (rc.x, rc.bottom - 2), (rc.right, rc.bottom - 2))
            elif k == "toggle":
                text(ctl.label, f, FG, (rc.x, rc.y + 5))
                sw = pygame.Rect(0, 0, int(38 * ui), int(20 * ui))
                sw.midright = (rc.right, rc.centery)
                on = bool(ctl.get())
                pygame.draw.rect(surface, ACCENT if on else BG[:3], sw, border_radius=sw.height // 2)
                pygame.draw.rect(surface, ACCENT if on else DIM, sw, 1, border_radius=sw.height // 2)
                knob = sw.height - 6
                pygame.draw.circle(surface, (0, 0, 0) if on else DIM,
                                   (sw.right - knob // 2 - 4 if on else sw.left + knob // 2 + 4, sw.centery), knob // 2)
            elif k == "stepper":
                text(ctl.label, f, FG, (rc.x, rc.y + 5))
                (minus, _), (plus, _) = r.parts
                for rect, sign in ((minus, "−"), (plus, "+")):
                    pygame.draw.rect(surface, LINE, rect, 1, border_radius=5)
                    text(sign, f, FG, rect.center, center=True)
                text(ctl.fmt.format(ctl.get()), f, ACCENT, ((minus.right + plus.left) // 2, rc.centery), center=True)
            elif k == "slider":
                val = ctl.get()
                text(ctl.label, f, FG, (rc.x, rc.y + 3))
                text(ctl.fmt.format(val), f, ACCENT, (rc.right, rc.y + 3), right=True)
                track = r.parts[0][0]
                mid = track.centery
                pygame.draw.line(surface, LINE, (track.x, mid), (track.right, mid), 3)
                t = (val - ctl.lo) / (ctl.hi - ctl.lo)
                pygame.draw.line(surface, ACCENT, (track.x, mid), (track.x + int(t * track.width), mid), 3)
                pygame.draw.circle(surface, ACCENT, (track.x + int(t * track.width), mid), int(6 * ui))
            elif k == "choice":
                text(ctl.label, f, FG, (rc.x, rc.y + 3))
                for (rect, payload), opt in zip(r.parts, ctl.options):
                    pill(rect, opt[1], ctl.get() == payload)
            elif k in ("chips", "presets"):
                text(ctl.label, f, FG, (rc.x, rc.y + 3))
                chosen = ctl.get()
                for rect, (label, payload) in r.parts:
                    on = (payload in chosen) if k == "chips" else (payload == chosen)
                    pill(rect, label, on)
            elif k == "text":
                text(ctl.label, f, FG, (rc.x, rc.y + 3))
                box = r.parts[0][0]
                focus = self.focus == "chars"
                pygame.draw.rect(surface, ACCENT if focus else LINE, box, 1, border_radius=5)
                val = ctl.get()
                shown = val + ("|" if focus else "")
                big = self._font(int(17 * ui))
                text(shown if shown else "click and type, or Ctrl+V", big if shown else fs, FG if shown else DIM, (box.x + 8, box.y + 4))
                # live preview in the chosen font, so what you type is what you get
                if val:
                    fname = self.cfg.glyph_font or "segoeuisymbol,segoeui,arial"
                    try:
                        pf = pygame.font.SysFont(fname, int(20 * ui))
                        img = pf.render(val[:24], True, ACCENT)
                        surface.blit(img, (rc.x, box.bottom + 2))
                    except Exception:
                        pass
            elif k == "fontlist":
                cur = ctl.get() or "(default)"
                text(f"{ctl.label}:  {cur}", f, FG, (rc.x, rc.y + 3))
                filt = r.parts[0][0]
                focus = self.focus == "fontfilter"
                pygame.draw.rect(surface, ACCENT if focus else LINE, filt, 1, border_radius=5)
                shown = self.font_filter + ("|" if focus else "")
                text(shown if shown else "search fonts (Wingdings, Symbol, ...)", f if shown else fs, FG if shown else DIM, (filt.x + 8, filt.y + 4))
                for rect, name in r.parts[1:]:
                    on = (name == cur)
                    if on:
                        pygame.draw.rect(surface, (40, 40, 40), rect)
                    text(name, fs, ACCENT if on else FG, (rect.x + 8, rect.y + 3))
            elif k == "button":
                pygame.draw.rect(surface, LINE, rc, 1, border_radius=6)
                text(ctl.label, f, FG, rc.center, center=True)
            elif k == "note":
                text(ctl.label, fs, DIM, (rc.x, rc.y + 8))
        surface.set_clip(clip)
