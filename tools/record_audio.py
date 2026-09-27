"""Record live system audio (mono float32) plus a now-playing timeline, so the
visualizer can be tested repeatably on real music (see render_live.py).

    python tools/record_audio.py rec 300     # -> rec.npy + rec.json (5 minutes)

Windows only (WASAPI loopback of the default output device).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from visualizer.now_playing import NowPlayingWatcher  # noqa: E402


def main() -> None:
    import pyaudiowpatch as pyaudio

    out, seconds = sys.argv[1], float(sys.argv[2])
    pa = pyaudio.PyAudio()
    dev = pa.get_default_wasapi_loopback()
    sr, ch = int(dev["defaultSampleRate"]), int(dev["maxInputChannels"])
    chunks: list[np.ndarray] = []
    received = [0]

    def cb(data, frames, info, status):
        chunks.append(np.frombuffer(data, dtype=np.float32).reshape(-1, ch).mean(axis=1).copy())
        received[0] += frames
        return (None, pyaudio.paContinue)

    stream = pa.open(format=pyaudio.paFloat32, channels=ch, rate=sr, input=True,
                     input_device_index=dev["index"], frames_per_buffer=512, stream_callback=cb)
    watcher = NowPlayingWatcher()
    watcher.start()
    timeline, last = [], None
    stream.start_stream()
    t0 = time.time()
    while time.time() - t0 < seconds:
        # loopback delivers nothing while the PC is silent: pad so the timeline stays true
        expected = int((time.time() - t0) * sr)
        if expected - received[0] > sr // 10:
            gap = expected - received[0]
            chunks.append(np.zeros(gap, dtype=np.float32))
            received[0] += gap
        lab = watcher.current().label()
        if lab != last:
            timeline.append({"sample": received[0], "label": lab})
            last = lab
        time.sleep(0.02)
    stream.stop_stream()
    stream.close()
    pa.terminate()
    watcher.stop()
    np.save(out + ".npy", np.concatenate(chunks).astype(np.float32))
    json.dump({"sample_rate": sr, "timeline": timeline}, open(out + ".json", "w"), indent=1)
    print(f"saved {received[0] / sr:.1f} s", [e["label"] for e in timeline])


if __name__ == "__main__":
    main()
