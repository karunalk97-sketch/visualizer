"""Indexed-color palettes and the quantization (with optional ordered dithering)
that maps a continuous grayscale intensity field down to a small palette.

This is the piece that gives the "8-bit / 4-bit / 1-bit" look: everything the
renderer draws is a single grayscale intensity per pixel (0.0-1.0); this module
decides which of a handful of colors that intensity becomes.
"""
from __future__ import annotations

import numpy as np

RGB = tuple[int, int, int]

# Built-in palettes, keyed by name. Each entry maps a bit depth to a list of
# colors ordered from "off" to "brightest".
PALETTES: dict[str, dict[int, list[RGB]]] = {
    "mono": {
        1: [(0, 0, 0), (255, 255, 255)],
        2: [(0, 0, 0), (85, 85, 85), (170, 170, 170), (255, 255, 255)],
        4: [(int(255 * i / 15),) * 3 for i in range(16)],
        8: [(int(255 * i / 255),) * 3 for i in range(256)],
    },
    "mono_green": {  # classic green CRT terminal
        1: [(0, 12, 0), (51, 255, 51)],
        2: [(0, 12, 0), (17, 85, 17), (34, 170, 34), (51, 255, 51)],
        4: [(int(51 * i / 15), int(12 + (255 - 12) * i / 15), int(51 * i / 15)) for i in range(16)],
        8: [(int(51 * i / 255), int(12 + (255 - 12) * i / 255), int(51 * i / 255)) for i in range(256)],
    },
    "gameboy": {  # DMG 4-shade green LCD
        1: [(15, 56, 15), (155, 188, 15)],
        2: [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
        4: [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)] * 4,  # repeats to fill 16
        8: [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)] * 64,
    },
    "ega16": {  # 16-color EGA-ish ramp for the 4-bit case, degrades gracefully otherwise
        1: [(0, 0, 0), (255, 255, 255)],
        2: [(0, 0, 0), (85, 0, 170), (0, 170, 170), (255, 255, 255)],
        4: [
            (0, 0, 0), (0, 0, 170), (0, 170, 0), (0, 170, 170),
            (170, 0, 0), (170, 0, 170), (170, 85, 0), (170, 170, 170),
            (85, 85, 85), (85, 85, 255), (85, 255, 85), (85, 255, 255),
            (255, 85, 85), (255, 85, 255), (255, 255, 85), (255, 255, 255),
        ],
        8: None,  # falls back to "mono" ramp at 8-bit, see get_palette()
    },
}

# 4x4 Bayer matrix, normalized to 0..1, used for ordered dithering.
_BAYER_4X4 = np.array(
    [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]],
    dtype=np.float32,
) / 16.0


def get_palette(name: str, bit_depth: int, custom_colors: list[list[int]] | None = None) -> list[RGB]:
    if name == "custom":
        colors = custom_colors or [(0, 0, 0), (255, 255, 255)]
        return [tuple(c) for c in colors]

    table = PALETTES.get(name, PALETTES["mono"])
    colors = table.get(bit_depth) or table.get(4) or PALETTES["mono"][4]
    return colors


def quantize(intensity: np.ndarray, colors: list[RGB], dither: bool) -> np.ndarray:
    """Map a (H, W) float array in [0, 1] to an (H, W, 3) uint8 RGB image
    using only the given palette, optionally with ordered dithering so
    low bit-depth palettes don't look as banded/flat.
    """
    n = len(colors)
    intensity = np.clip(intensity, 0.0, 1.0)

    if dither and n > 1:
        h, w = intensity.shape
        tile = np.tile(_BAYER_4X4, (h // 4 + 1, w // 4 + 1))[:h, :w]
        # Spread the dither threshold across one palette step, then quantize.
        step = 1.0 / (n - 1)
        intensity = intensity + (tile - 0.5) * step

    indices = np.clip(np.round(np.clip(intensity, 0.0, 1.0) * (n - 1)), 0, n - 1).astype(np.int32)
    palette_arr = np.array(colors, dtype=np.uint8)
    return palette_arr[indices]
