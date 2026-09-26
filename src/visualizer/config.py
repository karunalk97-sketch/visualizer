from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class Config:
    # Rendering -- always true black/white, only the number of gray levels changes.
    bit_depth: int = 1               # 1, 2, 3, or 4 (2, 4, 8, or 16 gray levels)

    fullscreen: bool = True
    pixel_size: int = 6              # fullscreen: screen_px / pixel_size = dither grid resolution
    grid_width: int = 160            # windowed/demo fallback grid size (fullscreen computes its own)
    grid_height: int = 90
    window_scale: int = 8            # windowed/demo upscale factor (nearest-neighbor -> chunky pixels)

    mode: str = "field"              # "field" (default: per-frequency spatial clusters) or "waveform"
    num_freq_points: int = 96        # how many frequency bins get their own spot on screen
    cluster_scale: int = 4           # cluster grid is this many dither-cells per side, coarser
    persistence: float = 0.88        # per-frame decay of the cluster energy (trailing/glow)
    decay: float = 0.85              # per-frame falloff of each frequency band's smoothed level
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
