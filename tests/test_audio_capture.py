import struct
import sys
import time

import numpy as np
import pytest

from visualizer.audio_capture import AppLoopbackSource, LatestWindow, RollingWindow, SyntheticSource, SystemTapSource


class FakeNative:
    """Stands in for proctap's ProcessLoopback."""

    def __init__(self, packets=(), bits=32, channels=2, rate=48000):
        self.packets = list(packets)
        self.started = self.stopped = False
        self._fmt = {"sample_rate": rate, "channels": channels, "bits_per_sample": bits}

    def get_format(self):
        return dict(self._fmt)

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def read(self):
        return self.packets.pop(0) if self.packets else b""


def f32(values):
    return np.asarray(values, dtype=np.float32).tobytes()


def test_rolling_window_keeps_the_latest_samples():
    w = RollingWindow(size=8)
    assert np.all(w.latest(4) == 0)
    w.push(np.arange(1, 6, dtype=np.float32))
    assert list(w.latest(5)) == [1, 2, 3, 4, 5]
    w.push(np.arange(6, 12, dtype=np.float32))            # more than fits: oldest fall off
    assert list(w.latest(8)) == [4, 5, 6, 7, 8, 9, 10, 11]
    w.push(np.arange(100, 130, dtype=np.float32))         # bigger than the whole window
    assert list(w.latest(3)) == [127, 128, 129]
    w.push(np.array([], dtype=np.float32))                # empty push is harmless
    w.push_silence(3)
    assert list(w.latest(3)) == [0, 0, 0]


def test_rolling_window_latest_is_a_copy():
    w = RollingWindow(size=4)
    w.push(np.ones(4, dtype=np.float32))
    w.latest(4)[:] = 9
    assert np.all(w.latest(4) == 1)


def test_app_source_reads_the_apps_audio_as_mono():
    native = FakeNative([f32([1.0, 0.0, 0.5, 0.5, 0.2, 0.4])])      # 3 stereo frames
    src = AppLoopbackSource(1234, "Spotify", native=native)
    chunk = next(src.frames(4))
    assert native.started and src.sample_rate == 48000 and src.pid == 1234
    assert np.allclose(chunk, [0.0, 0.5, 0.5, 0.3])                # last 4 mono samples: 0 pad, then the 3 frames averaged
    src.close()


def test_app_source_handles_16_bit_audio():
    pcm = struct.pack("<4h", 16384, -16384, 8192, 8192)             # 2 stereo frames
    src = AppLoopbackSource(1, "x", native=FakeNative([pcm], bits=16, rate=44100))
    chunk = next(src.frames(2))
    assert src.sample_rate == 44100
    assert np.allclose(chunk, [0.0, 0.25])


def test_app_source_ignores_a_torn_trailing_sample():
    src = AppLoopbackSource(1, "x", native=FakeNative([f32([0.2, 0.2, 0.9])]))   # 3 floats: one whole stereo frame + a stray
    assert np.allclose(next(src.frames(1)), [0.2])


def test_a_silent_app_drains_to_zero_instead_of_holding_the_last_sound():
    native = FakeNative([f32(np.full(2048, 0.8))])
    src = AppLoopbackSource(1, "x", native=native)
    src.IDLE_SECONDS = 0.01
    gen = src.frames(512)
    assert np.abs(next(gen)).max() > 0.5
    time.sleep(0.25)                                                # no more packets: the app went quiet
    for _ in range(3):
        last = next(gen)
    assert np.abs(last).max() < 0.05


def test_closing_the_app_source_stops_the_native_capture():
    native = FakeNative([])
    src = AppLoopbackSource(1, "x", native=native)
    gen = src.frames(64)
    next(gen)
    src.close()
    assert list(gen) == []                                          # the generator ends cleanly
    assert native.stopped


class FakeClock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_latest_window_always_returns_the_newest_audio_not_a_backlog():
    clock = FakeClock()
    w = LatestWindow(1000, size=64, clock=clock)
    for k in range(10):                                             # ten callbacks arrive before the frame reads
        w.push(np.full(8, k, dtype=np.float32))
    assert np.all(w.latest(8) == 9)                                 # the frame sees the newest, not the oldest
    assert list(w.latest(16)) == [8.0] * 8 + [9.0] * 8


def test_latest_window_pads_silence_when_capture_goes_quiet():
    clock = FakeClock()
    w = LatestWindow(1000, size=256, clock=clock)
    w.push(np.ones(200, dtype=np.float32))
    clock.t = 0.03                                                  # a short gap: still the music
    assert w.latest(10).max() == 1.0
    clock.t = 0.5                                                   # loopback sends nothing in silence
    assert np.all(w.latest(100) == 0.0)
    clock.t = 0.6
    w.push(np.full(5, 0.5, dtype=np.float32))                       # sound again
    assert list(w.latest(5)) == [0.5] * 5


FAKE_TAP = """
import struct, sys, threading, time
out = sys.stdout.buffer
out.write(b"VTAP" + struct.pack("<II", 1, 44100))
sys.stderr.write("hint: no-permission\\n"); sys.stderr.flush()
def stream():
    block = struct.pack("<3f", 0.25, 0.5, 0.75)
    while True:
        out.write(block[:5]); out.flush()                        # samples torn across writes
        out.write(block[5:]); out.flush()
        time.sleep(0.005)
threading.Thread(target=stream, daemon=True).start()
sys.stdin.read()                                                # runs until the app closes stdin
"""


def test_mac_system_tap_streams_the_helpers_audio(tmp_path):
    helper = tmp_path / "tap.py"
    helper.write_text(FAKE_TAP)
    src = SystemTapSource([sys.executable, str(helper)])
    gen = src.frames(3)
    deadline = time.monotonic() + 5
    chunk = next(gen)
    while not np.allclose(chunk, [0.25, 0.5, 0.75]) and time.monotonic() < deadline:
        chunk = next(gen)
    assert src.sample_rate == 44100 and np.allclose(chunk, [0.25, 0.5, 0.75])
    while not src.hint and time.monotonic() < deadline:
        time.sleep(0.01)
    assert "Privacy & Security" in src.hint                       # denied permission is explained
    src.close()
    assert src._proc.poll() is not None and list(gen) == []      # the helper quit and the stream ended


def test_mac_system_tap_raises_when_the_helper_cant_start(tmp_path):
    helper = tmp_path / "tap.py"
    helper.write_text("import sys; sys.stderr.write('error: needs macOS 14.2\\n'); sys.exit(4)")
    with pytest.raises(RuntimeError, match="needs macOS 14.2"):
        SystemTapSource([sys.executable, str(helper)])


def test_mac_falls_back_to_the_input_device_without_the_tap(monkeypatch):
    from visualizer import audio_router
    monkeypatch.setattr(audio_router.sys, "platform", "darwin")
    monkeypatch.setattr(audio_router, "SystemTapSource", lambda: (_ for _ in ()).throw(RuntimeError("old macOS")))
    monkeypatch.setattr(audio_router, "InputDeviceSource", lambda: "microphone")
    assert audio_router._default_system_factory() == "microphone"


def test_the_test_tone_source_produces_audio():
    clock = FakeClock()
    src = SyntheticSource(clock=clock)
    gen = src.frames(2048)
    clock.t = 1.0
    chunk = next(gen)
    assert chunk.shape == (2048,) and np.abs(chunk).max() > 0.1 and src.sample_rate == 48000
    clock.t = 1.5
    assert not np.allclose(next(gen), chunk)                        # it follows the clock: the latest window
