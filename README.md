# visualizer

A retro, bitmap-style system audio visualizer. It captures whatever's
currently playing on your Windows machine (no virtual audio cable needed),
runs an FFT, and renders it into a small low-resolution, indexed-color
framebuffer (1-bit, 2-bit, 4-bit, or 8-bit) that's scaled up with
nearest-neighbor scaling for a chunky pixel look.

## Architecture

```
audio_capture.py   WASAPI loopback capture (Windows) or a synthetic test tone
analyzer.py        FFT -> log-spaced band energies, with attack/decay smoothing
palette.py         Indexed color palettes + ordered (Bayer) dithering
renderer.py        Low-res framebuffer -> nearest-neighbor upscaled pygame window
config.py          JSON-backed settings (palette, bit depth, grid size, ...)
main.py            Glues it together, handles the window/event loop
```

Each piece is independent: the analyzer doesn't know about pygame, the
palette module doesn't know about audio, etc. That makes it straightforward
to add new visualization modes or new capture backends later.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install pyaudiowpatch
```

Run it against real system audio:

```
python -m visualizer.main
```

## Try it without Windows / without real audio

Every non-Windows environment (including this dev container) can run the
full rendering + analysis pipeline against a synthetic sweeping test tone:

```
pip install -e .
python -m visualizer.main --demo
```

## Controls

| Key | Effect |
|-----|--------|
| `P` | Cycle color palette |
| `B` | Cycle bit depth (1 / 2 / 4 / 8) |
| `D` | Toggle dithering |
| `M` | Toggle bars / waveform mode |
| `S` | Save current settings to `config.json` |
| `Esc` | Quit |

## Customization

Edit `config.json` (or press `S` in-app to persist your current live
settings):

- `bit_depth`: 1, 2, 4, or 8 — how many colors the palette is quantized to.
- `palette`: `mono`, `mono_green`, `gameboy`, `ega16`, or `custom`.
- `custom_colors`: list of `[r, g, b]` triples, used when `palette` is `custom`.
- `grid_width` / `grid_height`: the actual bitmap resolution being rendered.
- `window_scale`: integer upscale factor (this is what makes pixels chunky).
- `mode`: `bars` (spectrum analyzer) or `waveform`.
- `num_bands`, `decay`, `gain`: spectrum analyzer tuning.

## Tests

```
pip install pytest
pytest tests/
```

Tests cover the analyzer (band placement, decay behavior) and the palette
quantizer (bit-depth color counts, dithering output shape/range) — the parts
that don't need a real audio device or display to verify.

## Known limitation

`WasapiLoopbackSource` (real system audio) only works on Windows via
`pyaudiowpatch`, which wraps WASAPI loopback. macOS has no native loopback
API; supporting it later means capturing from a virtual audio driver like
BlackHole instead. Linux can use PulseAudio/PipeWire monitor sources. Both
are natural follow-ups — `audio_capture.py` is structured so adding another
`*Source` class is the only change needed.
