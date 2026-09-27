"""Perceptual audio features, measured in absolute terms.

Everything the picture reacts to is read here, from the latest window of audio,
in a way that means the same thing from song to song:

* loudness is decibels against a fixed reference (plus the Sensitivity setting),
  never re-normalised per song or per band, so the same loudness always looks
  the same and Sensitivity is a real threshold;
* transients (kick, bright hits) are detected as sudden rises in band energy, and
  their *strength* comes from how big the rise is and how loud the band actually
  is -- a limp kick in a quiet song stays small;
* timbre is read continuously: brightness (spectral centroid) and noisiness
  (spectral flatness) decide how sharp or grainy things look (bouba/kiki);
* sustained tonal energy (vocals, pads, chords) and the dominant pitch drive the
  flowing, continuous layers;
* "section" is a slow reading of energy relative to the song so far (calm verse
  vs. intense drop), which leans the whole vocabulary organic or stark.

Time constants are in seconds, so behaviour doesn't depend on the frame rate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

FFT_SIZE = 2048

# Calibration: dB -> 0..1. With the default Sensitivity (0.4x = -8 dB), a loud,
# loudness-normalised track (about -14 dBFS RMS, e.g. Spotify's default) reads
# about 0.65; quiet acoustic music (about -26 dBFS) about 0.4; silence 0.
LEVEL_FLOOR_DB = -50.0
LEVEL_RANGE_DB = 43.0
# Per-band (floor dB, range dB) for band power of the Hann-windowed FFT. Measured on
# real music (hip-hop, pop, R&B, film music at Spotify's normalised loudness): a typical
# moment reads about 0.6 in each band, the loudest ~5% about 0.9.
BAND_SCALE_DB = {"low": (-28.0, 46.0), "mid": (-22.0, 40.0), "high": (-34.0, 40.0)}

# The frequency field: five ranges, each drawn as its own shape, split into bands.
# (name, low Hz, high Hz, number of bands)
RANGES = (("bass", 40.0, 150.0, 5), ("lowmid", 150.0, 500.0, 8), ("vocals", 500.0, 2000.0, 10),
          ("highmid", 2000.0, 6000.0, 9), ("treble", 6000.0, 16000.0, 8))
RANGE_RELEASE = {"bass": 0.25, "lowmid": 0.2, "vocals": 0.25, "highmid": 0.14, "treble": 0.12}


def band_edges() -> tuple[np.ndarray, np.ndarray]:
    """(edges in Hz, range index of each band): log-spaced bands within each range."""
    edges, which = [], []
    for i, (_, lo, hi, n) in enumerate(RANGES):
        e = np.geomspace(lo, hi, n + 1)
        edges.extend(e[:-1] if i < len(RANGES) - 1 else e)
        which.extend([i] * n)
    return np.array(edges, dtype=np.float64), np.array(which, dtype=np.int64)


NUM_BANDS = sum(r[3] for r in RANGES)
# Per-band loudness of real music (median and loud 95th percentile, dB, at the default
# Sensitivity), measured on hip-hop, pop, R&B, film music, indie and house. A band at
# its typical level reads about 0.45, a loud moment about 0.9: absolute, never per song.
BAND_P50 = np.array([
    -12.4, -8.7, -8.7, -10.0, -10.2, -17.3, -17.7, -12.8, -19.9, -14.7, -16.3, -14.4, -15.9, -17.1, -15.6, -17.6,
    -16.8, -18.5, -18.8, -20.1, -20.8, -21.0, -21.0, -22.8, -22.5, -22.6, -23.5, -23.8, -25.8, -29.3, -29.3, -29.0,
    -27.3, -27.2, -28.2, -27.3, -29.9, -31.9, -35.0, -40.9])
BAND_P95 = np.array([
    8.2, 8.0, 8.0, 4.6, 3.7, -1.4, -1.9, 0.5, -4.5, -2.5, -2.6, -1.0, -0.4, -1.4, -0.5, -2.3,
    -2.6, -3.4, -4.4, -5.6, -5.8, -5.8, -6.5, -7.5, -7.9, -8.3, -8.8, -9.7, -10.1, -13.4, -14.1, -12.9,
    -11.6, -10.9, -11.4, -12.2, -13.4, -16.2, -18.1, -21.9])
_BAND_FLOOR = BAND_P50 - 18.0
_BAND_SPAN = (BAND_P95 - _BAND_FLOOR) / 0.9

LOW_BAND = (35.0, 160.0)
MID_BAND = (160.0, 2500.0)
HIGH_BAND = (2500.0, 16000.0)
PITCH_BAND = (80.0, 1000.0)


def _ema(prev: float, x: float, dt: float, tau: float) -> float:
    """Exponential moving average with a time constant in seconds."""
    if tau <= 0:
        return x
    a = 1.0 - math.exp(-dt / tau)
    return prev + (x - prev) * a


def _env(prev: float, x: float, dt: float, attack: float, release: float) -> float:
    """Envelope follower: rises with `attack`, falls with `release` (seconds)."""
    return _ema(prev, x, dt, attack if x > prev else release)


def _unit(db: float, floor: float, rng: float) -> float:
    return float(min(1.0, max(0.0, (db - floor) / rng)))


@dataclass
class Features:
    level: float = 0.0        # overall loudness 0..1 (absolute, after Sensitivity), smoothed
    low: float = 0.0          # bass loudness 0..1
    mid: float = 0.0          # body/vocal loudness 0..1
    high: float = 0.0         # treble loudness 0..1
    kick: float = 0.0         # a bass hit starting *this frame*: its strength 0..1, else 0
    hit: float = 0.0          # a bright transient (snare crack, hat, click) this frame: 0..1, else 0
    kick_env: float = 0.0     # decaying envelope of recent kicks (for continuous response)
    hit_env: float = 0.0      # decaying envelope of recent bright hits
    brightness: float = 0.0   # spectral centroid, 0 dull/round .. 1 bright/sharp
    noisiness: float = 0.0    # spectral flatness, 0 tonal .. 1 noise
    sustain: float = 0.0      # sustained tonal energy 0..1 (pads, vocals, chords), slow
    pitch: float = 0.5        # dominant pitch 0 (80 Hz) .. 1 (1 kHz), smoothed
    section: float = 0.0      # 0 calm .. 1 intense, slow (seconds)
    section_change: bool = False
    silent: bool = True
    bands: np.ndarray = field(default_factory=lambda: np.zeros(NUM_BANDS, dtype=np.float32))   # 0..1 per band, absolute
    waveform: np.ndarray = field(default_factory=lambda: np.zeros(FFT_SIZE, dtype=np.float32))


class AudioFeatures:
    """Call `update(window, dt)` once per frame with the latest window of audio
    (mono or multichannel float32, any length) and the frame time in seconds."""

    KICK_RISE_DB = 6.0          # bass must jump this far above its recent average to count as a hit
    HIT_RISE_DB = 6.0
    KICK_REFRACTORY = 0.12      # seconds: at most ~8 kicks a second
    HIT_REFRACTORY = 0.06
    SECTION_MIN_GAP = 8.0       # seconds between detected section changes

    def __init__(self, sample_rate: int = 48000, gain: float = 0.4) -> None:
        self.gain = gain
        self.sample_rate = sample_rate
        self._window = np.hanning(FFT_SIZE).astype(np.float32)
        self._win_norm = float((self._window ** 2).sum())
        self._freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)
        self._masks()
        self.f = Features()
        self._low_avg: float | None = None
        self._high_avg: float | None = None
        self._low_prev = self._high_prev = -120.0
        self._since_kick = 1e9
        self._since_hit = 1e9
        self._mid_prev = -120.0
        self._mid_var = 0.0
        self._sec_fast = 0.0
        self._sec_long = 0.0
        self._sec_anchor = 0.0
        self._sec_started = False
        self._since_section = 0.0
        self._t = 0.0
        self._pitch_log = 0.5

    def _masks(self) -> None:
        f = self._freqs
        self._low_m = (f >= LOW_BAND[0]) & (f < LOW_BAND[1])
        self._mid_m = (f >= MID_BAND[0]) & (f < MID_BAND[1])
        self._high_m = (f >= HIGH_BAND[0]) & (f < HIGH_BAND[1])
        self._pitch_m = (f >= PITCH_BAND[0]) & (f < PITCH_BAND[1])
        self._flat_m = (f >= 1000.0) & (f < 16000.0)
        self._cent_m = (f >= 40.0) & (f < 16000.0)
        edges, which = band_edges()
        self._band_lo = np.searchsorted(f, edges[:-1])
        self._band_hi = np.maximum(np.searchsorted(f, edges[1:]), self._band_lo + 1)   # at least one bin each
        self._band_release = np.array([RANGE_RELEASE[RANGES[i][0]] for i in which])

    def set_sample_rate(self, sample_rate: int) -> None:
        if sample_rate != self.sample_rate:
            self.sample_rate = sample_rate
            self._freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / sample_rate)
            self._masks()

    def reset_song(self) -> None:
        """A new track started: forget the previous song's section history."""
        self._sec_long = self._sec_fast
        self._sec_anchor = self._sec_fast
        self._since_section = 0.0

    def update(self, samples: np.ndarray, dt: float) -> Features:
        dt = float(min(max(dt, 1e-3), 0.25))
        self._t += dt
        f = self.f
        x = samples.mean(axis=1) if samples.ndim > 1 else samples
        x = np.asarray(x, dtype=np.float32) * np.float32(self.gain)
        if len(x) < FFT_SIZE:
            x = np.concatenate([np.zeros(FFT_SIZE - len(x), dtype=np.float32), x])
        x = x[-FFT_SIZE:]

        rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2)))
        db = 20.0 * math.log10(rms + 1e-9)
        level_now = _unit(db, LEVEL_FLOOR_DB, LEVEL_RANGE_DB)
        f.level = _env(f.level, level_now, dt, 0.03, 0.35)
        f.silent = db < LEVEL_FLOOR_DB + 2.0

        p = np.abs(np.fft.rfft(x * self._window)) ** 2 / self._win_norm
        low_db = 10.0 * math.log10(float(p[self._low_m].sum()) + 1e-12)
        mid_db = 10.0 * math.log10(float(p[self._mid_m].sum()) + 1e-12)
        high_db = 10.0 * math.log10(float(p[self._high_m].sum()) + 1e-12)
        # the frequency field: every band's own loudness, against real music's typical level
        cs = np.concatenate([[0.0], np.cumsum(p)])
        band_db = 10.0 * np.log10(cs[self._band_hi] - cs[self._band_lo] + 1e-12)
        now = np.clip((band_db - _BAND_FLOOR) / _BAND_SPAN, 0.0, 1.0).astype(np.float32)
        rel = np.exp(-dt / self._band_release).astype(np.float32)
        f.bands = np.maximum(now, f.bands * rel) if not f.silent else f.bands * rel   # snap up, ease down
        f.low = _env(f.low, _unit(low_db, *BAND_SCALE_DB["low"]), dt, 0.02, 0.25)
        f.mid = _env(f.mid, _unit(mid_db, *BAND_SCALE_DB["mid"]), dt, 0.04, 0.35)
        f.high = _env(f.high, _unit(high_db, *BAND_SCALE_DB["high"]), dt, 0.01, 0.15)

        # -- transients: a band jumping well above its own recent average (and still rising),
        # sized by how big the jump is *and* how loud the band really is
        self._since_kick += dt
        self._since_hit += dt
        f.kick, self._low_avg, self._low_prev, self._since_kick = self._onset(
            low_db, self._low_avg, self._low_prev, self._since_kick, dt, self.KICK_RISE_DB, self.KICK_REFRACTORY, f.low)
        f.hit, self._high_avg, self._high_prev, self._since_hit = self._onset(
            high_db, self._high_avg, self._high_prev, self._since_hit, dt, self.HIT_RISE_DB, self.HIT_REFRACTORY, f.high)
        f.kick_env = max(f.kick, f.kick_env * math.exp(-dt / 0.28))
        f.hit_env = max(f.hit, f.hit_env * math.exp(-dt / 0.14))

        # -- timbre
        cp = p[self._cent_m]
        total = float(cp.sum())
        if total > 1e-10 and not f.silent:
            centroid = float((cp * self._freqs[self._cent_m]).sum() / total)
            bright = min(1.0, max(0.0, math.log2(max(centroid, 1.0) / 200.0) / math.log2(4000.0 / 200.0)))
            fp = p[self._flat_m] + 1e-14
            flat = float(np.exp(np.mean(np.log(fp))) / np.mean(fp))
            noisy = min(1.0, max(0.0, (math.log10(flat + 1e-6) + 2.6) / 2.2))   # ~0.003 tonal .. ~0.5 noise
            f.brightness = _ema(f.brightness, bright, dt, 0.12)
            f.noisiness = _ema(f.noisiness, noisy, dt, 0.12)

        # -- sustained tonal energy: steady mids, not noise, not a stream of transients
        jump = abs(mid_db - self._mid_prev) if self._mid_prev > -100 else 0.0
        self._mid_prev = mid_db
        self._mid_var = _ema(self._mid_var, jump, dt, 0.25)
        steady = max(0.0, 1.0 - self._mid_var / 6.0)
        tonal = max(0.0, 1.0 - 1.2 * f.noisiness)
        sustained = min(1.0, 1.7 * f.mid * (0.35 + 0.65 * steady) * (0.4 + 0.6 * tonal))
        f.sustain = _env(f.sustain, sustained, dt, 0.35, 1.2)

        # -- dominant pitch (only when it is clear)
        pp = p[self._pitch_m]
        if pp.size and not f.silent:
            i = int(np.argmax(pp))
            conf = float(pp[i] / (pp.mean() + 1e-12))
            if conf > 6.0:
                hz = float(self._freqs[self._pitch_m][i])
                target = min(1.0, max(0.0, math.log2(hz / PITCH_BAND[0]) / math.log2(PITCH_BAND[1] / PITCH_BAND[0])))
                self._pitch_log = _ema(self._pitch_log, target, dt, 0.18)
        f.pitch = self._pitch_log

        # -- section: seconds-scale energy against the song so far, plus absolute intensity
        activity = 0.6 * f.level + 0.4 * min(1.0, f.kick_env + f.hit_env)
        if self._sec_started or not f.silent:
            if not self._sec_started:                      # start the history from the music, not from zero
                self._sec_fast = self._sec_long = self._sec_anchor = activity
                self._sec_started = True
            self._sec_fast = _ema(self._sec_fast, activity, dt, 3.0)
            self._sec_long = _ema(self._sec_long, activity, dt, 30.0)
        rel = min(1.0, max(0.0, 0.5 + 3.0 * (self._sec_fast - self._sec_long)))     # louder/busier than the song so far?
        absolute = min(1.0, max(0.0, (self._sec_fast - 0.3) / 0.5))                 # and how intense it is in itself
        f.section = 0.55 * rel + 0.45 * absolute
        self._since_section += dt
        f.section_change = False
        if self._since_section > self.SECTION_MIN_GAP and abs(self._sec_fast - self._sec_anchor) > 0.12:
            f.section_change = True
            self._sec_anchor = self._sec_fast
            self._since_section = 0.0

        f.waveform = x
        return f

    ONSET_AVG_SECONDS = 0.2

    def _onset(self, band_db: float, avg: float | None, prev: float, since: float, dt: float,
               rise_db: float, refractory: float, band_level: float) -> tuple[float, float, float, float]:
        """Returns (strength or 0, new average, new previous, new time-since-last)."""
        if avg is None:
            return 0.0, band_db, band_db, since
        rise = band_db - avg
        strength = 0.0
        if rise > rise_db and band_db > prev + 1.0 and since > refractory and band_level > 0.12:
            strength = min(1.0, 0.15 + (rise - rise_db) / 14.0) * min(1.0, 0.2 + 1.1 * band_level)
            strength = max(0.05, strength)
            since = 0.0
        avg = _ema(avg, band_db, dt, self.ONSET_AVG_SECONDS)
        return float(strength), avg, band_db, since
