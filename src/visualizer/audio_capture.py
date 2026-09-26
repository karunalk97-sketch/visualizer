"""Audio sources. WasapiLoopbackSource captures whatever is currently playing
on the default Windows output device (system audio, no virtual cable needed).
SyntheticSource generates a sweeping test tone so the rest of the pipeline can
be developed/demoed on any OS, including this one, without real hardware.
"""
from __future__ import annotations

import queue
import time
from typing import Iterator, Protocol

import numpy as np


class AudioSource(Protocol):
    sample_rate: int

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        ...


class SyntheticSource:
    """Generates a mix of tones sweeping across the audible range, useful for
    testing/demoing the visualizer without a real audio device or on non-Windows
    machines where WASAPI loopback isn't available.
    """

    def __init__(self, sample_rate: int = 48000, sweep_seconds: float = 8.0) -> None:
        self.sample_rate = sample_rate
        self.sweep_seconds = sweep_seconds
        self._t = 0.0

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        dt = 1.0 / self.sample_rate
        while True:
            t = self._t + np.arange(chunk_size) * dt
            phase = (t % self.sweep_seconds) / self.sweep_seconds
            freq = 80.0 * (200.0 ** phase)  # 80 Hz -> 16 kHz sweep
            bass = 0.6 * np.sin(2 * np.pi * 60.0 * t) * (0.5 + 0.5 * np.sin(2 * np.pi * 0.25 * t))
            sweep = 0.4 * np.sin(2 * np.pi * freq * t)
            samples = (bass + sweep).astype(np.float32)
            self._t += chunk_size * dt
            yield samples
            time.sleep(chunk_size / self.sample_rate)


class WasapiLoopbackSource:
    """Windows-only: captures the default output device's loopback stream via
    pyaudiowpatch (a WASAPI-patched fork of PyAudio). Import is deferred so the
    rest of the codebase stays importable on non-Windows machines.
    """

    def __init__(self) -> None:
        import pyaudiowpatch as pyaudio  # noqa: F401  (Windows-only dependency)

        self._pyaudio_module = pyaudio
        self._pa = pyaudio.PyAudio()
        device = self._pa.get_default_wasapi_loopback()
        self.sample_rate = int(device["defaultSampleRate"])
        self._device_index = device["index"]
        self._channels = device["maxInputChannels"]

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)

        def callback(in_data, frame_count, time_info, status):
            data = np.frombuffer(in_data, dtype=np.float32).reshape(-1, self._channels)
            try:
                q.put_nowait(data)
            except queue.Full:
                pass
            return (None, self._pyaudio_module.paContinue)

        stream = self._pa.open(
            format=self._pyaudio_module.paFloat32,
            channels=self._channels,
            rate=self.sample_rate,
            input=True,
            input_device_index=self._device_index,
            frames_per_buffer=chunk_size,
            stream_callback=callback,
        )
        stream.start_stream()
        try:
            while stream.is_active():
                # WASAPI loopback delivers nothing while the PC is silent; yield
                # silence instead of blocking so the window stays responsive.
                try:
                    yield q.get(timeout=0.05)
                except queue.Empty:
                    yield np.zeros((chunk_size, self._channels), dtype=np.float32)
        finally:
            stream.stop_stream()
            stream.close()

    def close(self) -> None:
        self._pa.terminate()


class InputDeviceSource:
    """macOS / Linux: reads an input device via sounddevice. macOS has no
    built-in loopback, so a virtual device (BlackHole, Loopback, Soundflower)
    is used if one is installed; otherwise the default input (the microphone)
    is used, which still reacts to music playing on the speakers."""

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

    def frames(self, chunk_size: int) -> Iterator[np.ndarray]:
        q: queue.Queue[np.ndarray] = queue.Queue(maxsize=8)

        def callback(indata, frame_count, time_info, status):
            try:
                q.put_nowait(indata.copy())
            except queue.Full:
                pass

        with self._sd.InputStream(
            device=self._device, channels=self._channels, samplerate=self.sample_rate,
            blocksize=chunk_size, dtype="float32", callback=callback,
        ):
            while True:
                try:
                    yield q.get(timeout=0.05)
                except queue.Empty:
                    yield np.zeros((chunk_size, self._channels), dtype=np.float32)

    def close(self) -> None:
        pass
