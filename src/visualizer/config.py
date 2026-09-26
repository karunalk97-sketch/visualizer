from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Config:
    # Rendering
    bit_depth: int = 2               # 1, 2, 4, or 8
    palette: str = "gameboy"         # see palette.PALETTES for built-ins
    custom_colors: list[list[int]] = field(default_factory=list)  # used when palette == "custom"
    dither: bool = True              # ordered (Bayer) dithering when quantizing to the palette

    grid_width: int = 64             # low-res framebuffer size (the "bitmap" resolution)
    grid_height: int = 32
    window_scale: int = 12           # integer upscale factor (nearest-neighbor -> chunky pixels)

    mode: str = "bars"               # "bars" or "waveform"
    num_bands: int = 32              # only used in "bars" mode
    decay: float = 0.85              # per-frame falloff for bar heights (0-1, higher = slower fall)
    gain: float = 1.0                # overall sensitivity multiplier

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
