"""Settings for one session. Nothing is saved: every launch starts from these
defaults, and the settings panel (or the Randomize button) changes them live."""
from __future__ import annotations

from dataclasses import dataclass, field

from .engines import default_elements

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

    mode: str = "field"              # "field" (the layered picture) or "waveform" (M toggles)
    overlap_invert: float = 1.0      # where layers overlap they invert (1 = white on white turns black; lower goes grey)
    gain: float = 0.4                # Sensitivity: fixed, calibrated on real music (not a setting any more)
    intensity: float = 1.0           # 0 = subtle (slow, sustained forms only) .. 1 = normal .. 2 = jarring (hits hit twice as hard)
    versions: list = field(default_factory=lambda: ["v7"])          # which versions draw; several = fused together
    elements: dict = field(default_factory=lambda: default_elements())   # per version: which of its elements are on
    pixel_decay: float = 0.0         # blends each frame with the last, so dithered pixels flicker less (0 = off)
    # Mix: turn each part of the spectrum up or down in the picture (0 = hide it, 2 = double)
    mix_bass: float = 1.0            # bass and kick: the ink blobs
    mix_lowmid: float = 1.0          # low mids (body, bass guitar, warmth): the orbs
    mix_vocals: float = 1.0          # vocal range (voice, lead melody): the rings
    mix_highmid: float = 1.0         # high mids (snare crack, presence): the stars
    mix_treble: float = 1.0          # treble (hats, cymbals, air): the sand
    reshuffle_on_new_song: bool = True

    render_mode: str = "pixels"      # "pixels" (dithered) or "chars" (glyphs; C toggles)
    glyph_cell: int = 12             # character mode: cell size in screen pixels
    glyph_mapping: str = "random"    # "random" glyph per cell, or "brightness" (denser glyph = brighter)
    glyph_sets: list = field(default_factory=lambda: ["shapes"])   # any of: shapes, symbols, ascii, binary

    show_now_playing: bool = True
    text_corner: str = "bottom_left"  # now-playing text corner; customization hint goes opposite

    fps: int = 60
