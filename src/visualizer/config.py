from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, fields
from pathlib import Path


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
    pixel_size: int = 6              # screen pixels per bitmap cell (bigger = chunkier)

    mode: str = "field"              # "field" (default: per-frequency random spots) or "waveform"
    num_freq_points: int = 96        # how many frequency bins get their own spots on screen
    spots_per_bin: int = 3           # random spots each frequency owns
    drift: float = 0.03              # how far spots wander from home (fraction of the screen)
    overlap_invert: float = 0.85     # where shapes overlap they invert (0 = just add, 1 = full negative)
    cluster_scale: int = 3           # cluster grid is this many dither-cells per side, coarser
    persistence: float = 0.88        # per-frame decay of the cluster energy (trailing/glow)
    decay: float = 0.85              # per-frame falloff of each frequency band's smoothed level
    gain: float = 1.0                # overall sensitivity multiplier
    reshuffle_on_new_song: bool = True

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
