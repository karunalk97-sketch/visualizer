"""Audio features against controlled fixtures: silence, steady tones, transients,
level changes, noise. The core promise: loudness is absolute (the same sound
always reads the same), and each descriptor responds to what it claims to."""
import numpy as np
import pytest

from visualizer.features import FFT_SIZE, AudioFeatures

SR = 48000
FPS = 60
HOP = SR // FPS


def tone(hz, seconds, amp=0.3):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * hz * t)).astype(np.float32)


def noise(seconds, amp=0.2, seed=0):
    return (amp * np.random.default_rng(seed).standard_normal(int(seconds * SR))).astype(np.float32)


def kicks(seconds, per_second=2.0, amp=0.8):
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    for s in np.arange(0.25, seconds, 1.0 / per_second):
        i, n = int(s * SR), int(0.25 * SR)
        tt = np.arange(min(n, len(t) - i)) / SR
        out[i:i + len(tt)] += amp * np.sin(2 * np.pi * (50 + 70 * np.exp(-tt * 30)) * tt) * np.exp(-tt * 12)
    return out.astype(np.float32)


def hats(seconds, per_second=4.0, amp=0.4, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    out = np.zeros_like(t)
    for s in np.arange(0.1, seconds, 1.0 / per_second):
        i, n = int(s * SR), int(0.05 * SR)
        m = min(n, len(t) - i)
        out[i:i + m] += amp * np.diff(rng.standard_normal(m + 1)) * np.exp(-np.arange(m) / SR * 90)
    return out.astype(np.float32)


def run(signal, feats=None, gain=0.4):
    """Feed a signal frame by frame like the app does; returns the feature snapshots."""
    feats = feats or AudioFeatures(SR, gain=gain)
    out = []
    for pos in range(FFT_SIZE, len(signal), HOP):
        f = feats.update(signal[pos - FFT_SIZE:pos], 1.0 / FPS)
        out.append((f.level, f.kick, f.hit, f.brightness, f.noisiness, f.sustain, f.pitch, f.section, f.section_change, f.silent))
    return feats, np.array(out, dtype=np.float64)


LEVEL, KICK, HIT, BRIGHT, NOISY, SUSTAIN, PITCH, SECTION, CHANGE, SILENT = range(10)


def test_silence_reads_silent_with_no_events():
    _, r = run(np.zeros(SR * 2, dtype=np.float32))
    assert r[:, LEVEL].max() == 0.0 and r[:, SILENT].all()
    assert r[:, KICK].max() == 0.0 and r[:, HIT].max() == 0.0


def test_loudness_is_absolute_the_same_sound_reads_the_same_whatever_came_before():
    steady = tone(220, 3, amp=0.1)
    _, fresh = run(steady)
    feats, _ = run(tone(220, 5, amp=0.6))                           # a loud song first...
    _, after = run(steady, feats)                                   # ...then the same quiet sound
    assert abs(fresh[-1, LEVEL] - after[-1, LEVEL]) < 0.01          # no per-song re-normalising


def test_louder_is_bigger_and_six_db_is_a_clear_step():
    levels = [run(tone(220, 2, amp=a))[1][-1, LEVEL] for a in (0.05, 0.1, 0.2)]
    assert levels[0] < levels[1] < levels[2]
    step = 6.0 / 43.0                                               # 6 dB on the calibrated scale
    assert abs((levels[1] - levels[0]) - step) < 0.03 and abs((levels[2] - levels[1]) - step) < 0.03


def test_sensitivity_is_a_real_threshold():
    quiet = tone(220, 2, amp=0.05)
    low = run(quiet, gain=0.4)[1][-1, LEVEL]
    high = run(quiet, gain=1.6)[1][-1, LEVEL]
    assert high - low == pytest.approx(20 * np.log10(4) / 43.0, abs=0.02)   # 4x sensitivity = +12 dB
    assert run(np.zeros(SR, np.float32), gain=3.0)[1][:, LEVEL].max() == 0.0   # but silence stays silent


def test_kicks_are_detected_at_their_real_rate_and_not_from_a_steady_tone():
    _, r = run(kicks(8, per_second=2.0))
    rate = (r[:, KICK] > 0).sum() / 8.0
    assert 1.5 <= rate <= 2.5
    _, steady = run(tone(60, 4, amp=0.5))
    assert (steady[:, KICK] > 0).sum() <= 1                         # only the very start may look like an onset


def test_a_harder_kick_is_a_stronger_event():
    soft = run(kicks(4, amp=0.2))[1][:, KICK]
    hard = run(kicks(4, amp=0.9))[1][:, KICK]
    assert hard[hard > 0].mean() > soft[soft > 0].mean()


def test_hats_are_bright_hits_not_kicks():
    _, r = run(hats(6))
    assert (r[:, HIT] > 0).sum() / 6.0 >= 2.5
    assert (r[:, KICK] > 0).sum() <= 2


def test_brightness_and_noisiness_follow_timbre():
    bass = run(tone(80, 2))[1][-1]
    hiss = run(noise(2))[1][-1]
    assert hiss[BRIGHT] > bass[BRIGHT] + 0.3                        # bright sound = sharp vocabulary
    assert hiss[NOISY] > bass[NOISY] + 0.3                          # noise = grain, tone = clean


def test_sustained_tonal_sound_reads_as_sustain_drums_do_not():
    chord = sum(tone(f, 4, amp=0.12) for f in (220, 277, 330, 440))
    pad = run(chord)[1][-60:, SUSTAIN].mean()
    drums = run(kicks(4) + hats(4))[1][-60:, SUSTAIN].mean()
    assert pad > 0.5 and pad > drums + 0.25


def test_pitch_rises_with_the_note():
    low = run(tone(110, 2))[1][-1, PITCH]
    high = run(tone(660, 2))[1][-1, PITCH]
    assert high > low + 0.4


def test_a_louder_section_raises_section_and_is_detected_as_a_change():
    verse = tone(220, 20, amp=0.03)
    drop = kicks(12, amp=0.9) + tone(220, 12, amp=0.3)
    _, r = run(np.concatenate([verse, drop]))
    in_verse = r[int(15 * FPS), SECTION]
    in_drop = r[-1, SECTION]
    assert in_drop > in_verse + 0.25
    assert r[int(20 * FPS):, CHANGE].any()


def test_stereo_mono_short_and_empty_windows_are_all_fine():
    stereo = np.stack([tone(220, 0.1), tone(220, 0.1)], axis=1)
    a = AudioFeatures(SR).update(stereo[-FFT_SIZE:], 1 / FPS)
    b = AudioFeatures(SR).update(stereo[-FFT_SIZE:].mean(axis=1), 1 / FPS)
    assert a.level == pytest.approx(b.level)
    f = AudioFeatures(SR)
    for chunk in (np.zeros(0, np.float32), np.ones(5, np.float32), np.zeros((0, 2), np.float32)):
        f.update(chunk, 1 / FPS)


def test_following_a_new_sample_rate():
    f = AudioFeatures(48000)
    f.set_sample_rate(44100)
    t = np.arange(FFT_SIZE) / 44100
    for _ in range(30):
        out = f.update((0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), 1 / FPS)
    assert f.sample_rate == 44100 and out.pitch > 0.5
