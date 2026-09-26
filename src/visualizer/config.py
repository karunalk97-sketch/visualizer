from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Rendering
    bit_depth: int = 1               # 1, 2, 4, or 8 -- default is true black/white
    palette: str = "mono"            # see palette.PALETTES for built-ins
    custom_colors: list[list[int]] = field(default_factory=list)  # used when palette == "custom"
    dither: bool = True              # ordered (Bayer) dithering when quantizing to the palette

    fullscreen: bool = True
    pixel_size: int = 6              # fullscreen: screen_px / pixel_size = grid resolution
    grid_width: int = 160            # windowed/demo fallback grid size (fullscreen computes its own)
    grid_height: int = 90
    window_scale: int = 8            # windowed/demo upscale factor (nearest-neighbor -> chunky pixels)

    mode: str = "field"              # "field" (full-screen halftone, default), "bars", or "waveform"
    num_bands: int = 32              # band count for "bars"; "field" uses one band per grid column
    decay: float = 0.85              # per-frame falloff for bar heights (0-1, higher = slower fall)
    gain: float = 1.0                # overall sensitivity multiplier

    show_now_playing: bool = True
    text_corner: str = "bottom_left"  # now-playing text corner; customization hint goes opposite

    fps: int = 60

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text())
        return cls(**{**asdict(cls()), **data})

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2))
