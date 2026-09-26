# visualizer

A full-screen, true black & white audio visualizer for whatever's currently
playing on your Windows machine (no virtual audio cable needed). Every
tracked frequency owns a fixed spot along a multi-turn spiral on screen (not
a grid, nothing mirrored); that spot only densifies when that frequency is
actually active. A bass-heavy track, a bright treble track, and a vocal-heavy
mix each light up a different region and carve out a different shape --
there's no fixed bar/level meter and no decorative animation layered on top,
every cluster you see is a direct trace of real frequency energy. The bottom
corners show the current track (pulled from Windows' now-playing info) and a
compact bit-depth readout.

## Architecture

```
audio_capture.py    WASAPI loopback capture (Windows) or a synthetic test tone
analyzer.py         FFT -> log-spaced frequency-bin energies, with attack/decay smoothing
spectral_field.py   Per-bin energies -> a persistent 2D cluster field (spiral layout, not a grid)
now_playing.py      Reads the current track from Windows System Media Transport Controls
palette.py          Black/white quantization at a chosen bit depth + ordered (Bayer) dithering
renderer.py         Low-res framebuffer -> nearest-neighbor upscaled to the real screen
config.py           JSON-backed settings (bit depth, grid size, ...)
main.py             Glues it together, fullscreen window + event loop
```

Each piece is independent: the analyzer doesn't know about pygame, the
palette module doesn't know about audio, etc.

## How the visual pattern works

1. `analyzer.py` turns each audio chunk into ~96 frequency-bin energies
   (log-spaced, so bass and treble both get meaningful resolution), with
   instant attack and slow decay so it doesn't flicker.
2. `spectral_field.py` gives each of those bins a *fixed* position along a
   spiral that winds through the screen several times, ordered by
   frequency -- so neighboring frequencies (which naturally light up
   together, since a single note's energy always spreads across nearby
   bins) stay spatially adjacent and form one connected cluster, while
   unrelated frequencies elsewhere on the spiral stay dark. Positions have a
   small amount of fixed jitter so nothing lines up in a perfect curve.
3. Each frame, active bins "splat" a soft blob at their spot, sized by how
   loud that frequency is; the whole field decays a little each frame so
   clusters glow and fade rather than snapping on/off.
4. That field is upscaled (blocky, nearest-neighbor) and run through
   black/white ordered dithering at the chosen bit depth, which is what
   produces the dot-density "8-bit dither" look.

Because the spiral position is fixed per frequency but which frequencies are
active depends entirely on the track, every song produces its own shape.

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
| `B` | Cycle bit depth (1 / 2 / 3 / 4 -- 2 / 4 / 8 / 16 gray levels, still black & white) |
| `M` | Toggle mode: frequency field / waveform |
| `N` | Toggle the now-playing + bit-depth text |
| `S` | Save current settings to `config.json` |
| `Esc` | Quit |

## Customization

Edit `config.json` (or press `S` in-app to persist your current live
settings):

- `bit_depth`: 1-4 -- how many gray levels the dithering uses (always black/white, never color).
- `fullscreen`: default `true`; use `--windowed` to override without editing the file.
- `pixel_size`: fullscreen dither resolution -- screen pixels per bitmap cell (bigger = chunkier).
- `num_freq_points`: how many frequency bins get their own spot on the spiral.
- `cluster_scale`: how coarse the underlying cluster grid is relative to the dither grid.
- `persistence`: how slowly clusters fade out (closer to 1 = longer glow/trail).
- `mode`: `field` (default) or `waveform`.
- `text_corner`: `bottom_left` or `bottom_right` for the now-playing text; the
  bit-depth readout automatically goes in the opposite corner.
- `show_now_playing`: toggle the bottom text overlay on/off entirely.
- `decay`, `gain`: spectrum analyzer tuning.
- `grid_width` / `grid_height` / `window_scale`: only used in `--windowed` mode (fullscreen derives its own grid from `pixel_size` and the real screen resolution).

## Tests

```
pip install pytest
pytest tests/
```

Tests cover the analyzer (band placement, decay behavior), the spectral
field (deterministic bin positions, 2D spread, energy decay, splat
locality), the now-playing label formatting, and the palette quantizer --
the parts that don't need a real audio device or display to verify.

## Known limitations

- `WasapiLoopbackSource` (real system audio) and `NowPlayingWatcher` (track
  metadata) only work on Windows, via `pyaudiowpatch` and `winsdk`
  respectively. macOS has no native audio loopback API (would need a virtual
  driver like BlackHole); Linux can use PulseAudio/PipeWire monitor sources.
  `audio_capture.py` is structured so adding another `*Source` class is the
  only change needed for a new platform.
- The now-playing text uses Windows' System Media Transport Controls, the
  same OS-level "what's playing" info shown on the lock screen -- it works
  automatically for any app that reports it (Spotify, browsers, most media
  players) with no per-app integration.
