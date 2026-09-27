# Design

> **Current look: the frequency field.** After trying layered "weather" (below),
> radial bursts and ink waves, the picture went back to the first version's idea --
> every frequency band owns a random spot, sized by how loud it is -- with each range
> drawn as its own shape (bass ink, low-mid orbs, vocal rings, high-mid stars,
> treble sand), absolute loudness against a per-band reference measured on real
> music, an area budget so nothing floods the screen, and a mix slider per range.
> The measurement principles below (absolute loudness, transients vs. sustain, no
> strobing, testing on real music) still apply; the layered vocabulary does not.

## Earlier direction: "Negative Weather"

A black-and-white, 1-bit dithered visualizer where the music is the only author.
Nothing on screen exists unless the sound put it there, and nothing stays that
the sound has stopped holding up.

## Philosophy

**Sound is measured the way it is heard, in absolute terms.** Loudness is read in
decibels against a fixed reference: the same loudness always looks the same,
across songs. Sensitivity is a real threshold you set once. No per-song
re-normalising, no per-band "loud for itself" tricks — those made every song
equally busy and the slider meaningless.

**Two kinds of time.** Sustained sound (vocals, pads, chords, the body of a mix)
drives *continuous* forms that flow, swell and ease: a terrain, waves, blobs,
fog. Transients (kicks, snares, hats) drive *events* that snap in and ease out:
blooms, ripples, sparkle, hairlines, small sharp bits. Continuous and punctual
never share an envelope, which is what lets the picture be calm and punchy at
once.

**Shape follows timbre, not frequency bins (bouba/kiki).** Dull, warm, low
sound is round; bright, harsh, noisy sound is sharp and fine. Sharpness is read
continuously from the spectral centroid and flatness. The sharp vocabulary is
deliberately small and short-lived, so harsh forms can accent but never dominate.

**Depth from layers, not lighting.** The far layer is sand: grains that gather
along the nodal lines of a slowly shifting field, finer as the pitch rises (a
Chladni plate, not symmetric). In front of it, a ridgeline terrain made of the
actual waveform recedes toward a horizon as a floating block, and each ridge
occludes the ones behind it (Unknown Pleasures, in perspective). The terrain
rises out of the dark when there is sustained sound and sinks away in sparse,
percussive passages, far ridges first. Other songs are led instead by a silk
ribbon of parallel strands that rides higher with the melody, or by ink --
warped metaballs that merge and pull apart. Everything in front inverts what it
overlaps (XOR), so depth also reads as negative space. There are no
full-screen flips.

**Crisp, never grey mush.** In 1-bit, grey dithers into checkerboard, so
nothing fades by greying: blooms dissolve into stable grains of sand, blobs and
shards shrink, hairlines retract, terrain ridges drop away. Solid white stays
solid (the dithering never puts stray dots in it).

**The song, the section and the intensity choose the vocabulary; the sound
chooses the moment.** Each song draws its own mix (how much terrain versus
waves versus blobs, bloom versus ripple, grain versus hairlines), its own
terrain geometry and its own layout, so no two songs look alike. The section
(a slow reading of energy relative to the song so far) leans calm passages
organic and intense passages stark. The strength of a hit escalates the
response: bloom, then bloom with ripple, then everything pulses. Sounds lean
toward their natural element, but only loosely.

**Restraint is part of the craft.** Coverage follows the music: sparse and dark
when it is quiet, full when it is loud, black in silence. Hits have a short
attack rather than a one-frame flash, large-area events are rate-limited, and
the number of simultaneous elements is capped: no strobing, no clutter.

## Mapping

| Measured (absolute)                   | Drives                                                    |
|---------------------------------------|-----------------------------------------------------------|
| Loudness (dBFS + sensitivity)         | Overall coverage and size of everything; silence = black  |
| Bass onset (flux in 40-160 Hz)        | Kick events: bloom / ripple / pulse, escalating with strength |
| Bright onset (flux above 2 kHz)       | Sparkle, hairlines, small sharp bits, edge crackle        |
| Spectral centroid ("brightness")      | How sharp the bright vocabulary gets (bouba/kiki)         |
| Spectral flatness ("noisiness")       | Grain versus clean lines; tonal sound feeds the terrain   |
| Sustained tonal energy                | Terrain presence, ribbon swell, ink size, sand thickness  |
| Dominant pitch                        | Height of the ribbon; how fine the sand pattern is        |
| Section energy (seconds, relative)    | Calm/organic versus intense/stark vocabulary; may hand the lead to another layer |
| Track change                          | New song mix, terrain geometry and layout                 |
| Waveform                              | The literal shape of each terrain ridge                   |

## How it was tuned

On real music captured from the system (hip-hop, Malayalam film music, pop,
French R&B, atmospheric indie, house, electro-swing, disco), not synthetic test
tones: `tools/record_audio.py` records playback with track names,
`tools/render_live.py` runs the exact app pipeline over it and writes screenshot
sequences, per-song contact sheets, frame cost, screen coverage and the largest
per-frame change (an anti-strobe check). Loudness scales and onset thresholds
were set from the measured distributions of those recordings.
