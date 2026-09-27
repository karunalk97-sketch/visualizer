# visualizer

A black & white audio visualizer for whatever is playing on your computer, with a
settings panel so you can change almost everything about how it looks.

- **Random shapes per frequency.** Every frequency owns a few random spots, and a
  spot only lights up when its frequency is actually active. Each spot has its own
  shape -- round blobs, spiky stars, long strings, spiky-and-stretched hybrids --
  pointing every which way and slowly spinning; louder means bigger and spikier.
  The layout is not a spiral, not sorted by pitch and not symmetric, and **every
  new song draws a brand new random layout**.
- **Soft ribbon waves that come and go.** When a synth, pad, chord or held vocal
  note *starts*, a soft ribbon is born, drifts across the screen and fades. Low,
  warm sounds make fat, slow swells (thick in the middle); rich, harmonic sounds
  make wobbly ribbons that are thicker towards the ends; bright sounds make thin,
  tight ones. Edges are feathered, never sharp. Drums and bass hits don't make
  waves, there are stretches with none, and in silence everything fades to black.
- **Everything inverts where it overlaps.** Where shapes cross, and where a ribbon
  crosses anything, the pixels invert like a negative: over solid white it cuts
  black, over gray it flips the gray, over black it shows up white -- a depth map
  across the whole picture.
- **Pixels or characters.** Draw the picture with dithered pixels, or with
  *characters*: built-in shapes (circle, square, triangle, diamond, plus, star,
  ...) and/or any text you type, in any font installed on your computer
  (Wingdings, Webdings, Symbol, ...). Each cell gets a random character, or a
  character chosen by brightness like classic ASCII art. Brightness sets how bright
  and how big each character is drawn, so bit depth still applies.
- **Quiet chrome.** Black title bar (Windows 11), and a slim status bar with the
  current track and display info in a normal system font.

![samples](docs/samples.png)

*Top: shapes + a ribbon, waves only, shapes only. Middle: character mode (built-in
shapes, Wingdings, brightness-mapped text). Bottom: the settings panel, real system
audio, and silence going to black.*

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

**Press `Tab` (or click the status bar) for the settings panel.** Everything changes
live and is saved when you close it:

- *Display*: fullscreen, bit depth, draw with pixels or characters, pixel size /
  character size
- *Characters*: random or by-brightness, built-in shapes on/off, your own characters
  (type or paste), font picker with search and a live preview, quick sets
- *Layers*: shapes on/off, waves on/off, overlap inversion, wave strength, wave
  softness, how often waves appear
- *Behaviour*: new layout on every song, status text, sensitivity

Hotkeys:

| Key | Effect |
|-----|--------|
| `Tab` | Open / close the settings panel |
| `1` / `2` / `3` | Toggle shapes / toggle waves / switch pixels <-> characters |
| `Up` / `Down` | Tighter / looser cells (pixels: 2, 3, 4, 6, 8, 12 px; characters: 6-40 px) |
| `B` | Cycle bit depth (1 / 2 / 3 / 4 -> 2 / 4 / 8 / 16 gray levels) |
| `F` / `F11` | Toggle fullscreen (`Esc` leaves fullscreen; it never closes the app) |
| `R` | Reshuffle to a new random layout right now |
| `M` | Toggle mode: frequency field / waveform |
| `N` | Toggle the now-playing + display text |
| `S` | Save settings now |

Settings live in `%APPDATA%\AudioVisualizer\config.json` on Windows and
`~/Library/Application Support/AudioVisualizer/` on macOS. On very large screens
the grid is coarsened automatically to keep the frame rate up.

## How it works

```
audio_capture.py    WASAPI loopback (Windows), input device (macOS/Linux), or a synthetic test tone
analyzer.py         FFT -> 96 log-spaced frequency energies with attack/decay smoothing
spectral_field.py   Random, drifting spots per frequency with varied shapes; XOR-style overlaps
waves.py            Soft ribbons born on note onsets from sustained tonal energy; inverts what is beneath
glyphs.py           Character mode: glyph atlas (built-in shapes + any font/characters), random or by-brightness
settings_ui.py      In-window settings panel (toggles, sliders, font list, text input)
palette.py          Black/white quantization + ordered (Bayer) dithering
renderer.py         Pixel path (nearest-neighbor upscale), character path, slim system-font status bar
now_playing.py      Current track from Windows media controls (winrt), used to spot song changes
window_style.py     Black title bar (Windows 11)
config.py           JSON settings
main.py             Resizable window, event loop
```

A new song is detected from the OS now-playing info (Windows) or, where that isn't
available, from music resuming after a silent gap. Ribbons are driven by each
frequency band's *sustained* energy (the lower of its current level and its recent
average), which is why held notes make waves and drum hits don't; a new ribbon is
born when a band's sustained energy starts, when the pitch moves, or occasionally
while a long note is held.

## Development

```
pip install -e ".[dev]"
pytest tests/
python tools/render_samples.py out/            # headless screenshots of every look
python -m PyInstaller --onefile --windowed --name AudioVisualizer --collect-submodules winrt --collect-all pyaudiowpatch --paths src packaging/entry.py
```

Pushing a tag like `v0.2.0` makes GitHub Actions build the Windows and macOS apps
and attach them to a release (`.github/workflows/build.yml`).

## Known limitations

- Real system-audio capture and track names are Windows-only; the macOS build
  hears audio through an input device and shows no track name. The macOS and
  Linux paths are untested on real hardware.
- Character mode is black and white: color emoji don't render (monochrome symbol
  fonts such as Wingdings and Segoe UI Symbol do).
- The black title bar needs Windows 11; Windows 10 gets a dark one, other
  platforms keep their normal title bar.
- The Windows exe is unsigned, so SmartScreen shows a warning on first run.
