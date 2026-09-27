"""Which applications are making sound right now (Windows).

Uses the Windows audio session list (via pycaw): every app that has opened an audio
stream shows up, flagged as playing or silent. Apps are grouped by executable, so
a browser with several processes is one entry, and the entry that is actually
playing wins. The list is polled on a background thread because COM calls can take
a few milliseconds and the picture must never stutter.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass
from typing import Callable, Iterable

# friendlier names for executables people actually recognise
KNOWN_LABELS = {
    "chrome": "Chrome", "msedge": "Edge", "firefox": "Firefox", "brave": "Brave", "opera": "Opera",
    "vivaldi": "Vivaldi", "spotify": "Spotify", "vlc": "VLC", "discord": "Discord", "steam": "Steam",
    "zoom": "Zoom", "teams": "Teams", "itunes": "iTunes", "musicbee": "MusicBee", "foobar2000": "foobar2000",
    "applemusic": "Apple Music", "wmplayer": "Windows Media Player", "youtubemusic": "YouTube Music",
    "tidal": "TIDAL", "amazonmusic": "Amazon Music", "deezer": "Deezer", "obs64": "OBS",
}
IGNORED = {"audiodg", "svchost", "python", "pythonw", "audiovisualizer"}   # system plumbing and ourselves


@dataclass(frozen=True)
class AudioApp:
    key: str        # lower-case executable name without .exe: stable across restarts
    pid: int
    label: str      # what the panel shows
    playing: bool


def label_for(process_name: str) -> str:
    stem = process_name[:-4] if process_name.lower().endswith(".exe") else process_name
    if stem.lower() in KNOWN_LABELS:
        return KNOWN_LABELS[stem.lower()]
    return stem.capitalize() if stem.islower() else stem


def collect_apps(sessions: Iterable[tuple[int, str, int]], own_pid: int | None = None) -> list[AudioApp]:
    """(pid, process name, state) triples -> one AudioApp per executable, playing
    ones first. State: 0 inactive, 1 active (playing), 2 expired."""
    own_pid = os.getpid() if own_pid is None else own_pid
    best: dict[str, AudioApp] = {}
    for pid, name, state in sessions:
        if not pid or pid == own_pid or not name:
            continue
        stem = name[:-4] if name.lower().endswith(".exe") else name
        key = stem.lower()
        if key in IGNORED or state == 2:
            continue
        app = AudioApp(key=key, pid=int(pid), label=label_for(name), playing=(state == 1))
        old = best.get(key)
        if old is None or (app.playing and not old.playing):
            best[key] = app
    return sorted(best.values(), key=lambda a: (not a.playing, a.label.lower()))


def _windows_sessions() -> list[tuple[int, str, int]]:
    import comtypes
    import psutil
    from pycaw.pycaw import AudioUtilities

    comtypes.CoInitialize()          # this runs on a worker thread
    try:
        out = []
        for s in AudioUtilities.GetAllSessions():
            pid = s.ProcessId
            if not pid:
                continue
            try:
                name = psutil.Process(pid).name()
            except Exception:
                continue
            out.append((pid, name, int(s.State)))
        return out
    finally:
        comtypes.CoUninitialize()


def per_app_supported() -> bool:
    """True when this machine can capture a single application's audio."""
    if sys.platform != "win32":
        return False
    try:
        import proctap._native  # noqa: F401
        import pycaw.pycaw  # noqa: F401
        import psutil  # noqa: F401
        return sys.getwindowsversion().build >= 19041     # process loopback needs Windows 10 2004+
    except Exception:
        return False


def list_audio_apps() -> list[AudioApp]:
    if not per_app_supported():
        return []
    try:
        return collect_apps(_windows_sessions())
    except Exception:
        return []


class AppLister:
    """Keeps a recent list of audio apps, refreshed on a background thread."""

    def __init__(self, fetch: Callable[[], list[AudioApp]] = list_audio_apps, min_interval: float = 1.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self._fetch = fetch
        self._min_interval = min_interval
        self._clock = clock
        self._apps: list[AudioApp] = []
        self._last = -1e9
        self._busy = threading.Lock()
        self._thread: threading.Thread | None = None

    @property
    def apps(self) -> list[AudioApp]:
        return list(self._apps)

    def request(self) -> None:
        """Start a refresh in the background unless one is running or the list is fresh."""
        now = self._clock()
        if now - self._last < self._min_interval or not self._busy.acquire(blocking=False):
            return
        self._last = now
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            self._apps = self._fetch()
        except Exception:
            pass
        finally:
            self._busy.release()

    def wait(self, timeout: float = 2.0) -> None:
        """For tests and start-up: block until the running refresh (if any) finishes."""
        t = self._thread
        if t is not None:
            t.join(timeout)
