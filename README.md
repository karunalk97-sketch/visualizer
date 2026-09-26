# visualizer

A black & white audio visualizer for whatever is playing on your computer. Every
frequency owns a few **random** spots on a dithered bitmap grid, and a spot only
lights up when its frequency is actually active. Each spot has its own shape --
round blobs, spiky stars, long strings, spiky-and-stretched hybrids -- pointing
every which way and slowly spinning; louder means bigger and spikier. **Where
shapes overlap they invert** (a negative, not just a brighter gray). The layout is not a spiral, not
sorted by pitch and not symmetric -- neighbouring frequencies land in unrelated
places -- and the spots slowly drift, so the picture feels organic instead of like
a meter. **Every new song draws a brand new random layout**, so no two songs look
alike. The bottom corners show the current track and the bit depth.

![samples](docs/samples.png)

*Bass-heavy, treble-bright, vocal and full-mix audio, the same mix with a different
song's layout, a capture of real system audio (a quiet moment), and the same mix
with overlaps simply adding (bottom) for comparison with the inverting default.*

## Get it running

**Easiest -- download the app** from the
[Releases page](../../releases/latest):

| OS | File | Notes |
|----|------|-------|
| Windows | `AudioVisualizer-windows.exe` | Double-click. Captures system audio directly (no cables). SmartScreen may warn because the app is unsigned: *More info -> Run anyway*. |
| macOS (Apple silicon) | `AudioVisualizer-mac.zip` | Unzip, then right-click the app -> *Open* the first time. macOS has no built-in loopback, so it listens to the default input (microphone), or to a virtual device like BlackHole if installed. |

**From source, one double-click** (installs Python packages for you on first run):

- Windows: double-click `Run-Windows.bat`
- macOS: double-click `Run-Mac.command`

Or by hand: `pip install -e .` then `python -m visualizer.main`
(add `--demo` for a built-in test tone).

## Using it

It opens as a normal window (minimize / maximize / **X**). It stays open when you
click into other apps or other monitors and only closes when you close it.

| Key | Effect |
|-----|--------|
| `F` / `F11` | Toggle fullscreen (`Esc` leaves fullscreen; it never closes the app) |
| `R` | Reshuffle to a new random layout right now |
| `B` | Cycle bit depth (1 / 2 / 3 / 4 -> 2 / 4 / 8 / 16 gray levels) |
| `M` | Toggle mode: frequency field / waveform |
| `N` | Toggle the now-playing + bit-depth text |
| `S` | Save current settings |

Settings are saved per user (`%APPDATA%\AudioVisualizer\config.json` on Windows,
`~/Library/Application Support/AudioVisualizer/` on macOS). Useful options:
`pixel_size` (chunkiness), `spots_per_bin` (how many spots each frequency owns),
`drift` (how far spots wander), `overlap_invert` (0 = overlaps add, 1 = full negative), `persistence` (glow/trail), `gain` (sensitivity),
`reshuffle_on_new_song`.

## How it works

```
audio_capture.py    WASAPI loopback (Windows), input device (macOS/Linux), or a synthetic test tone
analyzer.py         FFT -> 96 log-spaced frequency energies with attack/decay smoothing
spectral_field.py   Random, drifting spots per frequency with varied shapes; XOR-style overlaps; reshuffle on new song
now_playing.py      Current track from Windows media controls (winrt), used to spot song changes
palette.py          Black/white quantization + ordered (Bayer) dithering
renderer.py         Low-res framebuffer -> nearest-neighbor upscale to the window
config.py           JSON settings
main.py             Resizable window, event loop
```

A new song is detected from the OS now-playing info (Windows) or, where that isn't
available, from music resuming after a silent gap.

## Development

```
pip install -e ".[dev]"
pytest tests/
python tools/render_samples.py out/            # headless screenshots of several music styles
python -m PyInstaller --onefile --windowed --name AudioVisualizer --collect-submodules winrt --collect-all pyaudiowpatch --paths src packaging/entry.py
```

Pushing a tag like `v0.2.0` makes GitHub Actions build the Windows and macOS apps
and attach them to a release (`.github/workflows/build.yml`).

## Known limitations

- Real system-audio capture and track names are Windows-only; the macOS build
  hears audio through an input device and shows no track name. The macOS and
  Linux paths are untested on real hardware.
- The Windows exe is unsigned, so SmartScreen shows a warning on first run.
