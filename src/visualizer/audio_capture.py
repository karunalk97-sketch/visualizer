"""Audio sources.

* WasapiLoopbackSource   -- everything playing on the default Windows output device
                            (no virtual cable). Survives the output device changing
                            or disappearing: it re-opens instead of ending.
* AppLoopbackSource      -- ONE application's audio only (Windows 10 2004+ / 11),
                            via WASAPI process loopback, so the picture follows
                            Spotify or a browser instead of "whatever the PC is playing".
* SystemTapSource        -- macOS 14.2+: everything the Mac is playing, through a
                            Core Audio process tap (no virtual audio driver needed).
* InputDeviceSource      -- older macOS / Linux: an input device (the microphone, or
                            a virtual loopback device such as BlackHole).
* SyntheticSource        -- a sweeping test tone, for demos and tests.

Every source exposes `sample_rate` and `frames(chunk_size)`, a generator that yields
the *latest* `chunk_size` mono samples each time it is asked. It never blocks, never
builds up a backlog (so the picture never lags behind the music), and never ends on
its own (silence is yielded while nothing is playing).
"""
from __future__ import annotations

import os
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator, Protocol

import numpy as np


class AudioSource(Protocol):
    sample_rate: int

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        ...


class RollingWindow:
    """The most recent samples of a mono signal. Analysis reads the latest window
    every frame, so it works at any frame rate and any packet size."""

    def __init__(self, size: int = 8192) -> None:
        self.buf = np.zeros(size, dtype=np.float32)

    def push(self, mono: np.ndarray) -> None:
        n = len(mono)
        if n == 0:
            return
        size = len(self.buf)
        if n >= size:
            self.buf[:] = mono[-size:]
        else:
            self.buf[:-n] = self.buf[n:]
            self.buf[-n:] = mono

    def push_silence(self, n: int) -> None:
        if n > 0:
            self.push(np.zeros(min(n, len(self.buf)), dtype=np.float32))

    def latest(self, n: int) -> np.ndarray:
        return self.buf[-n:].copy()


class LatestWindow:
    """Thread-safe rolling window fed by an audio callback, read by the render loop.
    Reading always returns the newest audio, so a slow frame never makes the picture
    lag behind the music (a queue would pile up stale buffers). Capture APIs deliver
    nothing while the output is silent, so after a short gap it pads with silence."""

    GAP_SECONDS = 0.06

    def __init__(self, sample_rate: int, size: int = 16384, clock=time.monotonic) -> None:
        self.sample_rate = sample_rate
        self._win = RollingWindow(size)
        self._lock = threading.Lock()
        self._clock = clock
        self._last = self._padded = clock()

    def push(self, mono: np.ndarray) -> None:
        with self._lock:
            self._win.push(mono)
            self._last = self._clock()

    def latest(self, n: int) -> np.ndarray:
        now = self._clock()
        with self._lock:
            if now - self._last > self.GAP_SECONDS:
                start = max(self._last, self._padded)
                self._win.push_silence(int((now - start) * self.sample_rate))
                self._padded = now
            return self._win.latest(n)


class SyntheticSource:
    """Generates a mix of tones sweeping across the audible range, useful for
    testing/demoing the visualizer without a real audio device or on machines
    where no capture is available.
    """

    def __init__(self, sample_rate: int = 48000, sweep_seconds: float = 8.0, clock=time.monotonic) -> None:
        self.sample_rate = sample_rate
        self.sweep_seconds = sweep_seconds
        self._clock = clock
        self._t0 = clock()

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        dt = 1.0 / self.sample_rate
        while True:
            # the window that ends "now", like a live source
            end = self._clock() - self._t0
            t = end - (chunk_size - 1 - np.arange(chunk_size)) * dt
            phase = (t % self.sweep_seconds) / self.sweep_seconds
            freq = 80.0 * (200.0 ** phase)  # 80 Hz -> 16 kHz sweep
            bass = 0.6 * np.sin(2 * np.pi * 60.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t))
            beat = np.exp(-((t * 2.0) % 1.0) * 18.0)                     # a kick twice a second
            kick = 0.7 * np.sin(2 * np.pi * 55.0 * t) * beat
            sweep = 0.4 * np.sin(2 * np.pi * freq * t)
            yield (bass * 0.6 + kick + sweep).astype(np.float32)
            time.sleep(0.004)

    def close(self) -> None:
        pass


class WasapiLoopbackSource:
    """Windows-only: captures the default output device's loopback stream via
    pyaudiowpatch (a WASAPI-patched fork of PyAudio). Import is deferred so the
    rest of the codebase stays importable on non-Windows machines.

    If the default output device changes (headphones plugged in, output switched)
    or the stream dies, capture re-opens on the new default device instead of
    ending; while no device is available it yields silence.
    """

    POLL_SECONDS = 3.0

    def __init__(self) -> None:
        import pyaudiowpatch as pyaudio  # noqa: F401  (Windows-only dependency)

        self._pyaudio_module = pyaudio
        pa = pyaudio.PyAudio()
        try:
            device = pa.get_default_wasapi_loopback()   # fail early if there is no usable device
            self.sample_rate = int(device["defaultSampleRate"])
            self._device_key = (device["index"], device["name"])
        finally:
            pa.terminate()
        self._changed = threading.Event()
        self._closed = threading.Event()

    def _default_device(self):
        pa = self._pyaudio_module.PyAudio()
        try:
            return pa.get_default_wasapi_loopback()
        finally:
            pa.terminate()

    def _watch(self) -> None:
        """Flags a change of the default output device (PyAudio only sees devices
        that existed when it was created, so a fresh instance is needed)."""
        while not self._closed.wait(self.POLL_SECONDS):
            try:
                dev = self._default_device()
                if (dev["index"], dev["name"]) != self._device_key:
                    self._changed.set()
            except Exception:
                self._changed.set()   # device gone: try to re-open

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        threading.Thread(target=self._watch, daemon=True).start()
        pyaudio = self._pyaudio_module
        while not self._closed.is_set():
            pa = stream = None
            try:
                pa = pyaudio.PyAudio()
                device = pa.get_default_wasapi_loopback()
                channels = int(device["maxInputChannels"])
                self.sample_rate = int(device["defaultSampleRate"])
                self._device_key = (device["index"], device["name"])
                self._changed.clear()
                window = LatestWindow(self.sample_rate)

                def callback(in_data, frame_count, time_info, status, window=window, channels=channels):
                    window.push(np.frombuffer(in_data, dtype=np.float32).reshape(-1, channels).mean(axis=1))
                    return (None, pyaudio.paContinue)

                stream = pa.open(
                    format=pyaudio.paFloat32, channels=channels, rate=self.sample_rate, input=True,
                    input_device_index=device["index"], frames_per_buffer=512, stream_callback=callback,
                )
                stream.start_stream()
                while stream.is_active() and not self._changed.is_set() and not self._closed.is_set():
                    # always the newest audio: the picture never lags behind the music;
                    # silence (loopback sends nothing) is padded in, so this never blocks
                    yield window.latest(chunk_size)
            except GeneratorExit:
                raise
            except Exception:
                # No output device right now (or it just vanished): show silence, try again.
                for _ in range(25):
                    yield np.zeros(chunk_size, dtype=np.float32)
                    time.sleep(0.04)
            finally:
                try:
                    if stream is not None:
                        stream.stop_stream()
                        stream.close()
                except Exception:
                    pass
                try:
                    if pa is not None:
                        pa.terminate()
                except Exception:
                    pass

    def close(self) -> None:
        self._closed.set()


class AppLoopbackSource:
    """Windows 10 2004+ / 11: captures ONE application's audio (and the processes
    it starts) with WASAPI process loopback, so the picture follows Spotify or a
    browser rather than everything the PC is playing.

    `native` is the low-level capture object (proctap's ProcessLoopback); it is
    injectable so this can be tested without real audio.
    """

    IDLE_SECONDS = 0.06   # after this long with no packets, the app is treated as silent

    def __init__(self, pid: int, name: str = "", native=None) -> None:
        if native is None:
            from proctap._native import ProcessLoopback  # type: ignore[import-not-found]
            native = ProcessLoopback(pid)
        self.pid = pid
        self.name = name
        self._native = native
        fmt = native.get_format()
        self.sample_rate = int(fmt["sample_rate"])
        self._channels = max(1, int(fmt["channels"]))
        self._bits = int(fmt["bits_per_sample"])
        self._window = RollingWindow()
        self._closed = threading.Event()

    def _to_mono(self, data: bytes) -> np.ndarray:
        if self._bits == 16:
            a = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        else:
            a = np.frombuffer(data, dtype=np.float32)
        usable = len(a) // self._channels * self._channels
        return a[:usable].reshape(-1, self._channels).mean(axis=1)

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        self._native.start()
        last_data = last_pad = time.monotonic()
        try:
            while not self._closed.is_set():
                got = False
                while True:                                   # drain everything that has arrived
                    data = self._native.read()
                    if not data:
                        break
                    self._window.push(self._to_mono(data))
                    got = True
                now = time.monotonic()
                if got:
                    last_data = last_pad = now
                elif now - last_data > self.IDLE_SECONDS:     # nothing arriving: the app is silent, let the window drain to zero
                    self._window.push_silence(int((now - last_pad) * self.sample_rate))
                    last_pad = now
                yield self._window.latest(chunk_size)
                time.sleep(0.004)
        finally:
            try:
                self._native.stop()
            except Exception:
                pass

    def close(self) -> None:
        self._closed.set()


TAP_HELPER = "SystemAudioTap"
NO_PERMISSION_NOTE = ("Only silence is coming through while apps are playing: macOS is blocking it. "
                      "Allow Audio Visualizer in System Settings > Privacy & Security > "
                      "Screen & System Audio Recording, then reopen it.")


def find_tap_helper() -> Path:
    """The bundled macOS system-audio helper (built from packaging/mac/SystemAudioTap.swift)."""
    candidates = []
    if os.environ.get("VISUALIZER_TAP_HELPER"):
        candidates.append(Path(os.environ["VISUALIZER_TAP_HELPER"]))
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:                                                # inside the .app
        candidates += [Path(bundle) / TAP_HELPER, Path(bundle).parent / "Resources" / TAP_HELPER,
                       Path(sys.executable).parent / TAP_HELPER]
    candidates.append(Path(__file__).resolve().parents[2] / "build" / "mac" / TAP_HELPER)   # from source
    for path in candidates:
        if path.is_file() and os.access(path, os.X_OK):
            return path
    raise FileNotFoundError("the system audio helper isn't built")


class SystemTapSource:
    """macOS 14.2+: everything the Mac is playing, with nothing extra to install.

    The capture runs in a small bundled helper (a Core Audio process tap); it streams
    a 12-byte header ("VTAP", version, sample rate) and then mono float32 samples on
    stdout, and quits when its stdin closes. macOS asks once for permission to record
    system audio; if it is denied only silence arrives, and `hint` explains what to do.

    Raises if the helper is missing or can't start (older macOS), so the caller can
    fall back to the input device. `command` is injectable for tests.
    """

    MAGIC = b"VTAP"
    START_SECONDS = 1.5          # wait this long for it to fail; a permission prompt can take longer

    def __init__(self, command: list[str] | None = None) -> None:
        cmd = command or [str(find_tap_helper())]
        self.sample_rate = 48000
        self.hint = ""
        self._window = LatestWindow(self.sample_rate)
        self._started = threading.Event()
        self._ended = threading.Event()
        self._closed = threading.Event()
        self._errors: list[str] = []
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0)
        self._notes = threading.Thread(target=self._read_notes, daemon=True)
        self._notes.start()
        threading.Thread(target=self._read_audio, daemon=True).start()
        deadline = time.monotonic() + self.START_SECONDS
        while not (self._started.is_set() or self._ended.is_set()) and time.monotonic() < deadline:
            time.sleep(0.01)
        if self._ended.is_set() and not self._started.is_set():
            self._notes.join(1.0)
            self.close()
            raise RuntimeError(self._errors[-1] if self._errors else "the system audio helper stopped")

    def _read_notes(self) -> None:
        for raw in self._proc.stderr:
            line = raw.decode("utf-8", "replace").strip()
            if line == "hint: no-permission":
                self.hint = NO_PERMISSION_NOTE
            elif line.startswith("error: "):
                self._errors.append(line[len("error: "):])

    def _read_audio(self) -> None:
        out = self._proc.stdout
        try:
            header = b""
            while len(header) < 12:
                more = out.read(12 - len(header))
                if not more:
                    return
                header += more
            if header[:4] != self.MAGIC:
                self._errors.append("the system audio helper sent something unexpected")
                return
            _, rate = struct.unpack("<II", header[4:])
            self.sample_rate = self._window.sample_rate = int(rate)
            self._started.set()
            rest = b""
            while True:
                data = out.read(16384)
                if not data:
                    return
                data = rest + data
                usable = len(data) // 4 * 4
                self._window.push(np.frombuffer(data[:usable], dtype=np.float32))
                rest = data[usable:]
        finally:
            self._ended.set()

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        # ends when the helper does (e.g. the output device changed); the router re-opens
        while not self._closed.is_set() and not self._ended.is_set():
            yield self._window.latest(chunk_size)          # always the newest audio, never a backlog
            time.sleep(0.002)

    def close(self) -> None:
        self._closed.set()
        try:
            self._proc.stdin.close()                         # the helper's cue to quit
            self._proc.wait(1.0)
        except Exception:
            self._proc.kill()


class InputDeviceSource:
    """Older macOS / Linux: reads an input device via sounddevice. A virtual
    loopback device (BlackHole, Loopback, Soundflower) is used if one is installed;
    otherwise the default input (the microphone), which still reacts to music
    playing on the speakers."""

    PREFERRED = ("blackhole", "loopback", "soundflower", "monitor")

    def __init__(self) -> None:
        import sounddevice as sd

        self._sd = sd
        devices = sd.query_devices()
        chosen = None
        for idx, dev in enumerate(devices):
            if dev["max_input_channels"] > 0 and any(k in dev["name"].lower() for k in self.PREFERRED):
                chosen = idx
                break
        if chosen is None:
            chosen = sd.default.device[0]
        info = sd.query_devices(chosen)
        self._device = chosen
        self._channels = max(1, min(2, int(info["max_input_channels"])))
        self.sample_rate = int(info["default_samplerate"])
        self.device_name = info["name"]
        self._closed = threading.Event()

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        window = LatestWindow(self.sample_rate)

        def callback(indata, frame_count, time_info, status):
            window.push(indata.mean(axis=1).astype(np.float32))

        with self._sd.InputStream(
            device=self._device, channels=self._channels, samplerate=self.sample_rate,
            blocksize=512, dtype="float32", callback=callback,
        ):
            while not self._closed.is_set():
                yield window.latest(chunk_size)          # always the newest audio, never a backlog
                time.sleep(0.002)

    def close(self) -> None:
        self._closed.set()
