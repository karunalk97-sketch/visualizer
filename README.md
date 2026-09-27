# visualizer

A black & white, 1-bit audio visualizer for whatever is playing on your computer.
Pure code, no images: layers the music builds, calm when the music is calm and
punchy when it hits. The design (and why) is in [docs/DESIGN.md](docs/DESIGN.md).

- **Every version we built, selectable -- and fusable.** The *Versions* tab lists
  them all: **V1 Spots**, **V2 Shapes**, **V3 Adaptive**, **V4 Weather**,
  **V5 Bursts**, **V6 Ink waves** and **V7 Field** (the default). Pick one, or pick
  several and they are fused: drawn together, inverting where they overlap. Each
  version is broken down into its elements (terrain, ink, rings, stars, 3D
  depth...) so you can switch any of them off. **Randomize** shuffles across
  every version and element; **Randomize within these versions** keeps your pick.
- **Intensity** (Look tab) sets how subtle or jarring it is: at 0 only the slow,
  sustained forms move; at 2 hits hit twice as hard. Sensitivity is fixed,
  calibrated on real music.
- **The song's spectrum, scattered (V7).** About 40 frequency bands each own one spot,
  placed at random (a new layout every song). A spot is as big as its frequency is
  loud and shrinks away when it goes quiet, so the screen is a live picture of what
  the music is made of.
- **Each part of the sound has its own shape.** Bass is **ink** that melts together
  when it swells, low mids are **orbs**, the vocal range draws **hollow rings**,
  snares and high mids **soft stars**, hats and treble **sand**. Kicks make the ink
  swell a little extra; snares and hats make the stars and sand pop.
- **Mix it yourself.** The *Mix* tab has a slider per range (0-2x), so you can
  bring the bass, the vocals or the hats forward, or take any of them out.
- **Loudness is real.** Loudness is measured in absolute terms, so the same volume
  always looks the same. Quiet music stays sparse, loud music fills the screen,
  silence fades to black.
- **Everything inverts where it overlaps.** Where layers cross, the pixels invert
  like a negative -- depth from negative space, never a full-screen flash.
- **Pixels or characters.** Draw with dithered pixels, or with characters chosen
  from four sets you can mix: **Shapes**, **Symbols** (hearts, arrows, checks,
  flowers, moons, suns, bolts, notes...), **ASCII** and **Binary 0/1**. Each cell
  gets a random glyph, or one picked by brightness like classic ASCII art.
  Brightness sets how bright and how big each glyph is, so bit depth still applies.
- **Pick where the audio comes from.** Follow *all system audio*, or just **one
  application** -- Spotify, a browser, a media player, anything that is making
  sound -- so the picture isn't reacting to notifications or a video in another
  window. The status-bar title follows the app you chose.
- **Randomize.** One button that shuffles the whole look until you find one you like.
- **Session only.** Nothing is saved; every launch starts from the defaults.

![samples](docs/samples.png)

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
(add `--demo` for a built-in test tone, `--fullscreen` to start fullscreen, or
`--app spotify` to listen to one application from the start -- handy for a
desktop shortcut).

## Using it

It opens as a normal window (minimize / maximize / **X**). It stays open when you
click into other apps or other monitors and only closes when you close it.

**Press `Tab` (or click the status bar) for settings.** Six tabs, no scrolling:
*Look*, *Versions* (pick / fuse versions and switch their elements), *Mix* (one
slider per range -- bass, low mids, vocals, snare & high mids, hats & treble --
applied to every version), *Glyphs*, *Audio* and *Presets*.

- *Look*: pixels or characters, pixel / character size, bit depth,
  **intensity** (subtle - jarring), **decay** (blends each frame with the last
  so the dithered pixels flicker less -- 0 by default; turn it up for something
  softer and rounder), overlap inversion, fullscreen, new layout on every song,
  status text
- *Characters*: which sets to use (Shapes / Symbols / ASCII / Binary), random or
  by-brightness, character size
- *Audio*: **Listen to** all system audio or a single app. Apps that are making
  sound appear in the list (with *playing* / *silent*), newest state every second.
  If the app you picked isn't running yet, the picture waits and locks on when it
  starts; if capturing it fails you're told why and it falls back to system audio.
- *Presets*: click one to apply it. **Save** keeps the current look (versions,
  elements, mix, intensity, decay, inversion, pixels / characters, sizes, glyphs);
  **Copy a share code** puts a short `AVP1.` code on the clipboard to send to
  someone; **Paste a shared preset** applies a code from the clipboard and saves
  it. Presets live in `%LOCALAPPDATA%\AudioVisualizer\presets.json`, written only
  when you save or delete one. Fullscreen and window size aren't part of a preset.

The app opens with the **Default** preset: V1 Spots fused with V4 Weather, ASCII
characters by brightness at 8 px, 1-bit, intensity 2.0x, decay 0.11, bass 2.0 /
low mids 0.4 / vocals 1.9 / snare 0.4 / treble 2.0. Start with a different one
with `--preset "Preset 2"` (a saved name) or `--preset AVP1....` (a share code).

The **Randomize** button is always at the bottom of the panel.

| Key | Effect |
|-----|--------|
| `Tab` | Open / close the settings panel (`Esc` closes it; `Left` / `Right` switch tabs) |
| `Space` | Randomize the whole look |
| `C` | Switch between pixels and characters |
| `Up` / `Down` | Tighter / looser cells (pixels: 2-12 px; characters: 6-40 px) |
| `B` | Cycle bit depth (1 / 2 / 3 / 4 -> 2 / 4 / 8 / 16 gray levels) |
| `F` / `F11` | Toggle fullscreen (`Esc` leaves fullscreen; it never closes the app) |
| `R` | A new look for this song right now |
| `M` | Toggle mode: layered picture / plain waveform |
| `N` | Toggle the now-playing + display text |

On very large screens (1080p and up), and when several versions are fused, the
pixel cells grow a little automatically to keep the frame rate up; the status bar
shows the size in use. Fusing all seven versions at once runs at roughly 25-30 fps.

## How it works

```
audio_capture.py    System loopback (survives output-device changes), one-app capture, input device, test tone
audio_apps.py       Which apps are making sound right now (Windows audio sessions)
audio_router.py     Chooses / switches the audio source at runtime; waits, falls back, never blocks
features.py         Absolute loudness, kick / bright-hit onsets, brightness, noisiness, sustain, pitch, section
engines/            One module per version (v1_spots ... v7_field) and fusion.py, which runs any mix of them
imaging.py          Fast bilinear resize and seamless value noise for the scene
glyphs.py           Character mode: glyph atlas (Shapes / Symbols / ASCII / Binary), random or by-brightness
randomizer.py       The Randomize button
settings_ui.py      In-window tabbed settings panel
palette.py          Black/white quantization + ordered (Bayer) dithering
renderer.py         Pixel path (nearest-neighbor upscale), character path, slim system-font status bar
now_playing.py      Current track from Windows media controls (winrt), used to spot song changes
window_style.py     Black title bar (Windows 11)
config.py           Session settings (never saved)
main.py             Resizable window, event loop
```

Every frame the latest ~43 ms of audio is measured in absolute terms (decibels
against a fixed reference, plus Sensitivity -- never re-normalised per song).
Transients (a band jumping well above its recent average) become events that
snap in and ease out; sustained, tonal sound drives the continuous layers;
brightness and noisiness decide how sharp or grainy things get. See
[docs/DESIGN.md](docs/DESIGN.md) for the full mapping.

A new song is detected from the OS now-playing info (Windows) or, where that isn't
available, from music resuming after a silent gap.

## Development

```
pip install -e ".[dev]"
pytest tests/
python tools/record_audio.py rec 300           # record 5 min of what's playing (Windows), with track names
python tools/render_live.py out/ --audio rec   # run the real pipeline over it: screenshots, contact sheets, stats
python tools/render_samples.py out/            # the same from synthetic test music (no audio device needed)
python -m PyInstaller --onefile --windowed --name AudioVisualizer --collect-submodules winrt --collect-all pyaudiowpatch --paths src packaging/entry.py
```

Pushing a tag like `v0.2.0` makes GitHub Actions build the Windows and macOS apps
and attach them to a release (`.github/workflows/build.yml`).

## Known limitations

- Real system-audio capture, single-app capture and track names are Windows-only
  (single-app needs Windows 10 version 2004 or newer). The macOS build hears audio
  through an input device, can't pick a single app, and shows no track name.
  Single-app capture on macOS would need ScreenCaptureKit (macOS 13+) and Screen
  Recording permission. The macOS and Linux paths are untested on real hardware.
- For a browser, the *browser's* audio is captured (all its tabs), not one tab.
- Character mode is black and white, drawn from the four built-in sets (no custom
  fonts or typed text).
- The black title bar needs Windows 11; Windows 10 gets a dark one, other
  platforms keep their normal title bar.
- The Windows exe is unsigned, so SmartScreen shows a warning on first run.
- Desktop only for now: a browser version (for hosting online) is a separate future step.
