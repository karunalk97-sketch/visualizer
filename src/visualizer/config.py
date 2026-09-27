"""Settings for one session. Nothing is saved: every launch starts from these
defaults, and the settings panel (or the Randomize button) changes them live."""
from __future__ import annotations

from dataclasses import dataclass, field

PIXEL_STEPS = [2, 3, 4, 6, 8, 12]        # screen pixels per bitmap cell, tightest first
GLYPH_CELL_STEPS = list(range(6, 41, 2))  # character cell sizes
GLYPH_SETS = [("shapes", "Shapes"), ("symbols", "Symbols"), ("ascii", "ASCII"), ("binary", "Binary 0/1")]


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
    waves: bool = True               # one smooth foam line sweeping across, shaped by the music (2 toggles)
    wave_strength: float = 0.6       # how strongly the foam line inverts what is under it (0..1)
    wave_softness: float = 0.6       # 0 = crisper edges, 1 = very feathered
    wave_rate: float = 1.0           # how often waves appear (2 = twice as often)
    depth: float = 0.35              # relief lighting that makes the picture read as 3D (0 = flat)
    reshuffle_on_new_song: bool = True

    render_mode: str = "pixels"      # "pixels" (dithered) or "chars" (glyphs; 3 toggles)
    glyph_cell: int = 12             # character mode: cell size in screen pixels
    glyph_mapping: str = "random"    # "random" glyph per cell, or "brightness" (denser glyph = brighter)
    glyph_sets: list = field(default_factory=lambda: ["shapes"])   # any of: shapes, symbols, ascii, binary

    show_now_playing: bool = True
    text_corner: str = "bottom_left"  # now-playing text corner; customization hint goes opposite

    fps: int = 60
