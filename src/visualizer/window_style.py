"""Makes the window's title bar black so it disappears into the picture.

Windows 11 lets an app pick the title bar colour through DWM (build 22000+);
on Windows 10 only the dark title bar is honoured, and other platforms keep
their normal chrome. Every call is best-effort and never raises.
"""
from __future__ import annotations

import sys

import pygame

_DWMWA_USE_IMMERSIVE_DARK_MODE = 20
_DWMWA_BORDER_COLOR = 34
_DWMWA_CAPTION_COLOR = 35
_DWMWA_TEXT_COLOR = 36


def blacken_title_bar() -> None:
    """Call after every pygame.display.set_mode (SDL may recreate the window)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        hwnd = pygame.display.get_wm_info().get("window")
        if not hwnd:
            return
        dwm = ctypes.windll.dwmapi
        dwm.DwmSetWindowAttribute.argtypes = [ctypes.c_void_p, ctypes.c_uint, ctypes.c_void_p, ctypes.c_uint]

        def set_attr(attr: int, value: int) -> None:
            v = ctypes.c_int(value)
            dwm.DwmSetWindowAttribute(hwnd, attr, ctypes.byref(v), ctypes.sizeof(v))

        set_attr(_DWMWA_USE_IMMERSIVE_DARK_MODE, 1)
        set_attr(_DWMWA_CAPTION_COLOR, 0x000000)  # COLORREF is 0x00BBGGRR
        set_attr(_DWMWA_BORDER_COLOR, 0x000000)
        set_attr(_DWMWA_TEXT_COLOR, 0x505050)     # title text present but barely there
    except Exception:
        pass
