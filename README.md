# visualizer

A full-screen, true black & white halftone/dither audio visualizer for
whatever's currently playing on your Windows machine (no virtual audio
cable needed). Every column of the screen maps to its own frequency band,
bars rise from the bottom, and an animated ambient texture keeps the whole
screen alive rather than just a strip of bars. The bottom corners show the
current track (pulled from Windows' now-playing info) and a compact
customization readout.

## Architecture

```
audio_capture.py   WASAPI loopback capture (Windows) or a synthetic test tone
analyzer.py        FFT -> log-spaced band energies, with attack/decay smoothing
fields.py          Per-column band energies -> full-screen intensity field
now_playing.py     Reads the current track from Windows System Media Transport Controls
palette.py         Indexed color palettes + ordered (Bayer) dithering
renderer.py        Low-res framebuffer -> nearest-neighbor upscaled to the real screen
config.py          JSON-backed settings (palette, bit depth, grid size, ...)
main.py            Glues it together, fullscreen window + event loop
```

Each piece is independent: the analyzer doesn't know about pygame, the
palette module doesn't know about audio, etc. That makes it straightforward
to add new visualization modes or new capture backends later.

## Setup (Windows)

```
python -m venv .venv
.venv\Scripts\activate
pip install -e .
pip install pyaudiowpatch winsdk
```

Run it against real system audio (opens fullscreen by default):

```
python -m visualizer.main
```

Use `--windowed` the first time if you want to confirm it works before going
fullscreen:

```
python -m visualizer.main --windowed
```

## Try it without Windows / without real audio

Every non-Windows environment (including this dev container) can run the
full rendering + analysis pipeline against a synthetic sweeping test tone,
and the now-playing text automatically falls back to blank on non-Windows:

```
pip install -e .
python -m visualizer.main --demo --windowed
```

## Controls

| Key | Effect |
|-----|--------|
| `P` | Cycle color palette |
| `B` | Cycle bit depth (1 / 2 / 4 / 8) |
| `D` | Toggle dithering |
| `M` | Cycle mode: full-screen field / bars / waveform |
| `N` | Toggle the now-playing + customization text |
| `S` | Save current settings to `config.json` |
| `Esc` | Quit |

## Customization

Edit `config.json` (or press `S` in-app to persist your current live
settings):

- `bit_depth`: 1, 2, 4, or 8 — how many colors the palette is quantized to (default 1 = true black/white).
- `palette`: `mono`, `mono_green`, `gameboy`, `ega16`, or `custom`.
- `custom_colors`: list of `[r, g, b]` triples, used when `palette` is `custom`.
- `fullscreen`: default `true`; use `--windowed` to override without editing the file.
- `pixel_size`: fullscreen dither resolution — screen pixels per bitmap cell (bigger = chunkier).
- `mode`: `field` (default, full-screen halftone), `bars`, or `waveform`.
- `text_corner`: `bottom_left` or `bottom_right` for the now-playing text; the
  customization readout automatically goes in the opposite corner.
- `show_now_playing`: toggle the bottom text overlay on/off entirely.
- `num_bands`, `decay`, `gain`: spectrum analyzer tuning.
- `grid_width` / `grid_height` / `window_scale`: only used in `--windowed` mode (fullscreen derives its own grid from `pixel_size` and the real screen resolution).

## Tests

```
pip install pytest
pytest tests/
```

Tests cover the analyzer (band placement, decay behavior), the field builder
(per-column mapping, bars rising from the bottom), the now-playing label
formatting, and the palette quantizer — the parts that don't need a real
audio device or display to verify.

## Known limitations

- `WasapiLoopbackSource` (real system audio) and `NowPlayingWatcher` (track
  metadata) only work on Windows, via `pyaudiowpatch` and `winsdk`
  respectively. macOS has no native audio loopback API (would need a virtual
  driver like BlackHole); Linux can use PulseAudio/PipeWire monitor sources.
  `audio_capture.py` is structured so adding another `*Source` class is the
  only change needed for a new platform.
- The now-playing text uses Windows' System Media Transport Controls, the
  same OS-level "what's playing" info shown on the lock screen — it works
  automatically for any app that reports it (Spotify, browsers, most media
  players) with no per-app integration.
