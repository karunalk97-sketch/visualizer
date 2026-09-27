"""Presets: save a look you like, bring it back, and share it as a short code.

A preset is the whole look -- which versions are fused and their elements, the mix,
intensity, decay, overlap inversion, pixels or characters, sizes, bit depth, glyph
sets. Window things (fullscreen, window size) are not part of a preset.

Settings themselves still last for the session only; presets are the one thing
saved, and only when you press Save. A share code is plain text ("AVP1....") you can
paste into a chat or a note; pasting it back (or `--preset CODE`) applies it.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import zlib
from pathlib import Path
from typing import Callable

from .config import GLYPH_CELL_STEPS, GLYPH_SETS, PIXEL_STEPS, Config
from .engines import REGISTRY, elements_of

CODE_PREFIX = "AVP1."
MIX_FIELDS = ("mix_bass", "mix_lowmid", "mix_vocals", "mix_highmid", "mix_treble")
_RANGES = {"intensity": (0.0, 2.0), "pixel_decay": (0.0, 0.9), "overlap_invert": (0.0, 1.0), **{m: (0.0, 2.0) for m in MIX_FIELDS}}
FIELDS = ("render_mode", "pixel_size", "glyph_cell", "bit_depth", "intensity", "pixel_decay", "overlap_invert",
          "glyph_sets", "glyph_mapping", "versions", "elements", *MIX_FIELDS, "reshuffle_on_new_song", "show_now_playing")

# The look the app opens with: V1 Spots fused with V4 Weather, drawn in ASCII by
# brightness, bass/vocals/treble pushed up, low mids and snare pulled back, jarring.
DEFAULT_PRESET = {
    "render_mode": "chars", "glyph_cell": 8, "bit_depth": 1, "intensity": 2.0, "pixel_decay": 0.11,
    "overlap_invert": 1.0, "glyph_sets": ["ascii"], "glyph_mapping": "brightness",
    "versions": ["v1", "v4"],
    "mix_bass": 2.0, "mix_lowmid": 0.4, "mix_vocals": 1.9, "mix_highmid": 0.4, "mix_treble": 2.0,
}
BUILTIN = {"default": ("Default", DEFAULT_PRESET), "field": ("V7 Field (clean)", {})}


def snapshot(cfg: Config) -> dict:
    """The current look as plain data."""
    out = {}
    for k in FIELDS:
        v = getattr(cfg, k)
        out[k] = {kk: list(vv) for kk, vv in v.items()} if isinstance(v, dict) else list(v) if isinstance(v, list) else v
    return out


def apply(cfg: Config, preset: dict) -> None:
    """Apply a preset onto a Config: every value is checked, anything unknown or
    out of range is ignored, anything missing falls back to the normal default."""
    base = Config()
    for k in FIELDS:
        setattr(cfg, k, getattr(base, k))
    for k, v in (preset or {}).items():
        if k not in FIELDS:
            continue
        try:
            if k in _RANGES:
                lo, hi = _RANGES[k]
                setattr(cfg, k, round(min(hi, max(lo, float(v))), 3))
            elif k == "render_mode" and v in ("pixels", "chars"):
                cfg.render_mode = v
            elif k == "pixel_size" and int(v) in PIXEL_STEPS:
                cfg.pixel_size = int(v)
            elif k == "glyph_cell" and int(v) in GLYPH_CELL_STEPS:
                cfg.glyph_cell = int(v)
            elif k == "bit_depth" and int(v) in (1, 2, 3, 4):
                cfg.bit_depth = int(v)
            elif k == "glyph_mapping" and v in ("random", "brightness"):
                cfg.glyph_mapping = v
            elif k == "glyph_sets":
                sets = [s for s, _ in GLYPH_SETS if s in v]
                cfg.glyph_sets = sets or cfg.glyph_sets
            elif k == "versions":
                vs = [x for x in REGISTRY if x in v]
                cfg.versions = vs or cfg.versions
            elif k == "elements" and isinstance(v, dict):
                for ver, els in v.items():
                    if ver in REGISTRY:
                        keep = [e for e, _ in elements_of(ver) if e in els]
                        if keep:
                            cfg.elements[ver] = keep
            elif k in ("reshuffle_on_new_song", "show_now_playing"):
                setattr(cfg, k, bool(v))
        except (TypeError, ValueError):
            continue


def encode(preset: dict) -> str:
    raw = json.dumps(preset, separators=(",", ":"), sort_keys=True).encode()
    return CODE_PREFIX + base64.urlsafe_b64encode(zlib.compress(raw, 9)).decode().rstrip("=")


def decode(code: str) -> dict:
    """The preset in a share code; raises ValueError if it isn't one."""
    code = "".join(str(code).split())
    start = code.find(CODE_PREFIX)
    if start < 0:
        raise ValueError("not a preset code")
    body = code[start + len(CODE_PREFIX):]
    try:
        data = json.loads(zlib.decompress(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4))))
    except Exception as exc:
        raise ValueError("damaged preset code") from exc
    if not isinstance(data, dict):
        raise ValueError("not a preset code")
    return data


def presets_path() -> Path:
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return base / "AudioVisualizer" / "presets.json"


class PresetStore:
    """Your saved presets, in one small JSON file (written only when you save or delete)."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = path or presets_path()

    def load(self) -> list[dict]:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return [p for p in data if isinstance(p, dict) and isinstance(p.get("name"), str) and isinstance(p.get("settings"), dict)]
        except (OSError, ValueError, TypeError):
            return []

    def _write(self, items: list[dict]) -> bool:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(items, indent=1), encoding="utf-8")
            return True
        except OSError:
            return False

    def add(self, settings: dict, name: str | None = None) -> str | None:
        items = self.load()
        taken = {p["name"] for p in items}
        n = len(items) + 1
        while name is None or name in taken:
            name, n = f"Preset {n}", n + 1
        items.append({"name": name, "settings": settings})
        return name if self._write(items) else None

    def delete(self, name: str) -> bool:
        items = self.load()
        kept = [p for p in items if p["name"] != name]
        return len(kept) != len(items) and self._write(kept)


def _win_clipboard():
    import ctypes
    from ctypes import wintypes
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    user32.OpenClipboard.argtypes, user32.OpenClipboard.restype = [wintypes.HWND], wintypes.BOOL
    user32.GetClipboardData.argtypes, user32.GetClipboardData.restype = [wintypes.UINT], ctypes.c_void_p
    user32.SetClipboardData.argtypes, user32.SetClipboardData.restype = [wintypes.UINT, ctypes.c_void_p], ctypes.c_void_p
    kernel32.GlobalAlloc.argtypes, kernel32.GlobalAlloc.restype = [wintypes.UINT, ctypes.c_size_t], ctypes.c_void_p
    kernel32.GlobalLock.argtypes, kernel32.GlobalLock.restype = [ctypes.c_void_p], ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [ctypes.c_void_p]
    return ctypes, user32, kernel32


def _clipboard_get() -> str:
    try:
        if sys.platform == "win32":
            ctypes, user32, kernel32 = _win_clipboard()
            if not user32.OpenClipboard(None):
                return ""
            try:
                handle = user32.GetClipboardData(13)             # CF_UNICODETEXT
                if not handle:
                    return ""
                ptr = kernel32.GlobalLock(handle)
                try:
                    return ctypes.wstring_at(ptr) if ptr else ""
                finally:
                    kernel32.GlobalUnlock(handle)
            finally:
                user32.CloseClipboard()
        import subprocess
        cmd = ["pbpaste"] if sys.platform == "darwin" else ["xclip", "-selection", "clipboard", "-o"]
        return subprocess.run(cmd, capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return ""


def _clipboard_put(text: str) -> bool:
    try:
        if sys.platform == "win32":
            ctypes, user32, kernel32 = _win_clipboard()
            data = text.encode("utf-16-le") + b"\x00\x00"
            if not user32.OpenClipboard(None):
                return False
            try:
                user32.EmptyClipboard()
                handle = kernel32.GlobalAlloc(0x0002, len(data))   # GMEM_MOVEABLE; the clipboard owns it afterwards
                ptr = kernel32.GlobalLock(handle)
                ctypes.memmove(ptr, data, len(data))
                kernel32.GlobalUnlock(handle)
                return bool(user32.SetClipboardData(13, handle))
            finally:
                user32.CloseClipboard()
        import subprocess
        cmd = ["pbcopy"] if sys.platform == "darwin" else ["xclip", "-selection", "clipboard"]
        return subprocess.run(cmd, input=text, text=True, timeout=2).returncode == 0
    except Exception:
        return False


class PresetManager:
    """What the settings panel's Presets tab talks to."""

    MAX_SHOWN = 6                  # the built-ins plus your most recent presets

    def __init__(self, cfg: Config, on_applied: Callable[[], None] = lambda: None, store: PresetStore | None = None,
                 clip_get: Callable[[], str] = _clipboard_get, clip_put: Callable[[str], bool] = _clipboard_put) -> None:
        self.cfg = cfg
        self.store = store or PresetStore()
        self.on_applied = on_applied
        self._get, self._put = clip_get, clip_put
        self.selected = "b:default"
        self.message = ""

    def options(self) -> list[tuple[str, str, str]]:
        """(key, name, note) for the list: built-ins first, then your newest presets."""
        out = [("b:" + k, name, "built in") for k, (name, _) in BUILTIN.items()]
        mine = self.store.load()[-(self.MAX_SHOWN - len(out)):]
        out += [("u:" + p["name"], p["name"], "saved") for p in reversed(mine)]
        return out

    def _settings(self, key: str) -> dict | None:
        if key.startswith("b:") and key[2:] in BUILTIN:
            return BUILTIN[key[2:]][1]
        if key.startswith("u:"):
            return next((p["settings"] for p in self.store.load() if p["name"] == key[2:]), None)
        return None

    def choose(self, key: str) -> None:
        settings = self._settings(key)
        if settings is None:
            return
        apply(self.cfg, settings)
        self.selected = key
        self.message = f"Applied {key[2:] if key.startswith('u:') else BUILTIN[key[2:]][0]}."
        self.on_applied()

    def save(self) -> None:
        name = self.store.add(snapshot(self.cfg))
        if name:
            self.selected = "u:" + name
            self.message = f"Saved as {name}."
        else:
            self.message = "Couldn't save the preset (the folder isn't writable)."

    def share(self) -> None:
        code = encode(snapshot(self.cfg))
        self.message = ("Share code copied. Paste it anywhere; whoever pastes it back gets this exact look."
                        if self._put(code) else f"Couldn't reach the clipboard. Your code: {code}")

    def paste(self) -> None:
        try:
            settings = decode(self._get())
        except ValueError:
            self.message = "The clipboard doesn't hold a preset code. Copy one (it starts with AVP1.) and try again."
            return
        apply(self.cfg, settings)
        name = self.store.add(snapshot(self.cfg), None)
        self.selected = "u:" + name if name else self.selected
        self.message = f"Applied the shared preset{' and saved it as ' + name if name else ''}."
        self.on_applied()

    def can_delete(self) -> bool:
        return self.selected.startswith("u:")

    def delete(self) -> None:
        if self.can_delete() and self.store.delete(self.selected[2:]):
            self.message = f"Deleted {self.selected[2:]}."
            self.selected = "b:default"
