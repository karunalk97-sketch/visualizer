# visualizer

A black & white audio visualizer for whatever is playing on your computer.

- **Random shapes per frequency.** Every frequency owns a few random spots on a
  dithered bitmap grid, and a spot only lights up when its frequency is actually
  active. Each spot has its own shape -- round blobs, spiky stars, long strings,
  spiky-and-stretched hybrids -- pointing every which way and slowly spinning;
  louder means bigger and spikier. The layout is not a spiral, not sorted by pitch
  and not symmetric, and **every new song draws a brand new random layout**.
- **Waves for synths and harmonies.** Sustained tonal sound -- synth pads, chords,
  held vocal notes -- sends fine wave lines cascading across the screen. Higher
  notes make tighter waves, louder ones make stronger, faster waves. Drums and bass
  hits don't make waves, and in a lull or silence they fade away completely.
- **Everything inverts where it overlaps.** Where shapes cross, and where a wave
  crosses anything, the pixels invert like a negative: a wave over solid white cuts
  a black line, over gray it flips the gray, over black it shows up white. The
  result reads as a depth map across the whole picture.
- **Quiet chrome.** Black title bar (Windows 11), and a slim status bar with the
  current track and the display settings in a normal system font.

![samples](docs/samples.png)

*Synth pad and vocal (waves across shapes), pad + drums + bass, drums and bass alone
(no waves), a capture of real system audio, and the pad after the music stops
(black).*

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
| `Up` / `Down` | Tighter / looser pixel grid (2, 3, 4, 6, 8, 12 px per cell; default 3) |
| `B` | Cycle bit depth (1 / 2 / 3 / 4 -> 2 / 4 / 8 / 16 gray levels) |
| `W` | Toggle the waves |
| `R` | Reshuffle to a new random layout right now |
| `M` | Toggle mode: frequency field / waveform |
| `N` | Toggle the now-playing + display text |
| `S` | Save current settings |

The status bar shows the track on the left and the current bit depth and pixel size
on the right. On very large screens the grid is coarsened automatically to keep the
frame rate up.

Settings are saved per user (`%APPDATA%\AudioVisualizer\config.json` on Windows,
`~/Library/Application Support/AudioVisualizer/` on macOS). Useful options:
`pixel_size` (grid tightness), `spots_per_bin` (how many spots each frequency owns),
`drift` (how far spots wander), `overlap_invert` (0 = overlaps add, 1 = full
negative), `waves` and `wave_strength`, `persistence` (glow/trail), `gain`
(sensitivity), `reshuffle_on_new_song`.

## How it works

```
audio_capture.py    WASAPI loopback (Windows), input device (macOS/Linux), or a synthetic test tone
analyzer.py         FFT -> 96 log-spaced frequency energies with attack/decay smoothing
spectral_field.py   Random, drifting spots per frequency with varied shapes; XOR-style overlaps; reshuffle on new song
waves.py            Cascading waves from sustained tonal energy; inverts whatever is beneath them
now_playing.py      Current track from Windows media controls (winrt), used to spot song changes
palette.py          Black/white quantization + ordered (Bayer) dithering
renderer.py         Low-res framebuffer -> nearest-neighbor upscale, plus the slim system-font status bar
window_style.py     Black title bar (Windows 11)
config.py           JSON settings
main.py             Resizable window, event loop
```

A new song is detected from the OS now-playing info (Windows) or, where that isn't
available, from music resuming after a silent gap. Waves are driven by each
frequency band's *sustained* energy (the lower of its current level and its recent
average), which is why held notes make waves and drum hits don't.

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
- The black title bar needs Windows 11; Windows 10 gets a dark one, other
  platforms keep their normal title bar.
- The Windows exe is unsigned, so SmartScreen shows a warning on first run.
