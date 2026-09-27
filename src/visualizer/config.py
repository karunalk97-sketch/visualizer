from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


PIXEL_STEPS = [2, 3, 4, 6, 8, 12]        # screen pixels per bitmap cell, tightest first
GLYPH_CELL_STEPS = list(range(6, 41, 2))  # character cell sizes


def default_config_path() -> Path:
    """Per-user settings location, so the packaged app can save settings even
    though its own folder is read-only or temporary."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home()))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "AudioVisualizer" / "config.json"


@dataclass
class Config:
    # Rendering -- always true black/white, only the number of gray levels changes.
    bit_depth: int = 1               # 1, 2, 3, or 4 (2, 4, 8, or 16 gray levels)

    fullscreen: bool = False         # opens as a normal window (with an X); F toggles fullscreen
    window_width: int = 1280
    window_height: int = 720
    pixel_size: int = 3              # screen pixels per bitmap cell: smaller = tighter/finer (Up/Down keys)

    mode: str = "field"              # "field" (default: per-frequency random spots) or "waveform"
    num_freq_points: int = 96        # how many frequency bins get their own spots on screen
    spots_per_bin: int = 3           # random spots each frequency owns
    drift: float = 0.03              # how far spots wander from home (fraction of the screen)
    overlap_invert: float = 0.85     # where shapes overlap they invert (0 = just add, 1 = full negative)
    persistence: float = 0.88        # per-frame decay of the cluster energy (trailing/glow)
    decay: float = 0.85              # per-frame falloff of each frequency band's smoothed level
    gain: float = 1.0                # overall sensitivity multiplier
    show_shapes: bool = True         # the spiky/blobby shapes (1 toggles)
    waves: bool = True               # soft ribbon waves born from synths/harmonies (2 toggles)
    wave_strength: float = 0.6       # how strongly waves invert what is under them (0..1)
    wave_softness: float = 0.6       # 0 = crisper ribbon edges, 1 = very feathered
    wave_rate: float = 1.0           # how often waves appear (2 = twice as often)
    reshuffle_on_new_song: bool = True

    render_mode: str = "pixels"      # "pixels" (dithered) or "chars" (glyphs; 3 toggles)
    glyph_cell: int = 12             # character mode: cell size in screen pixels
    glyph_mapping: str = "random"    # "random" glyph per cell, or "brightness" (denser glyph = brighter)
    glyph_font: str = ""             # font for your characters ("" = default symbol-capable font)
    glyph_chars: str = ""            # your own characters, any script or symbol font
    glyph_shapes: list = field(default_factory=lambda: ["circle", "square", "triangle", "diamond", "plus"])

    show_now_playing: bool = True
    text_corner: str = "bottom_left"  # now-playing text corner; customization hint goes opposite

    fps: int = 60

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        try:
            data = json.loads(p.read_text())
        except (OSError, ValueError):
            return cls()
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(asdict(self), indent=2))
