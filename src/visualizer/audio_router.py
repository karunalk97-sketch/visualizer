"""Chooses where the audio comes from, and lets you change it while the app runs.

* "All system audio"  -- whatever the PC is playing (or the input device on macOS/Linux).
* One application     -- only that app's audio (Spotify, a browser, a player...).

The choice is remembered by executable name, not process id, so it survives the app
restarting: if the chosen app isn't running or has no audio yet, the picture waits
(silent) and locks on the moment it appears. If capturing an app fails, the router
falls back to system audio and says why. `read()` never blocks and never raises.
"""
from __future__ import annotations

import sys
import time
from typing import Callable

import numpy as np

from .audio_apps import AppLister, AudioApp, per_app_supported
from .audio_capture import AppLoopbackSource, InputDeviceSource, SyntheticSource, WasapiLoopbackSource
from .features import FFT_SIZE

SYSTEM = "system"
APP_PREFIX = "app:"


def _default_system_factory():
    if sys.platform == "win32":
        return WasapiLoopbackSource()
    return InputDeviceSource()


def _pid_exists(pid: int) -> bool:
    try:
        import psutil
        return psutil.pid_exists(pid)
    except Exception:
        return True


class AudioRouter:
    MAINTAIN_SECONDS = 1.0

    def __init__(
        self,
        demo: bool = False,
        system_factory: Callable[[], object] | None = None,
        app_factory: Callable[[int, str], object] | None = None,
        lister: AppLister | None = None,
        clock: Callable[[], float] = time.monotonic,
        pid_exists: Callable[[int], bool] = _pid_exists,
        supported: bool | None = None,
    ) -> None:
        self._demo = demo
        self._system_factory = system_factory or _default_system_factory
        self._app_factory = app_factory or (lambda pid, name: AppLoopbackSource(pid, name))
        self._lister = lister or AppLister()
        self._clock = clock
        self._pid_exists = pid_exists
        self.supported = per_app_supported() if supported is None else supported
        self.selected = SYSTEM
        self.notice = ""                       # why we fell back / what went wrong, for the panel
        self._source = None
        self._gen = None
        self._is_demo = False
        self._wanted: str | None = None        # executable key of the app we are waiting for / capturing
        self._wanted_label = ""
        self._bound_pid: int | None = None
        self._rate = 48000
        self._last_maintain = -1e9
        self._chunk = FFT_SIZE                 # the window the picture analyses every frame
        self._closed = False
        self._open_system()

    # -- sources -----------------------------------------------------------------

    def _close_source(self) -> None:
        gen, src = self._gen, self._source
        self._gen = self._source = None
        self._bound_pid = None
        try:
            if src is not None and hasattr(src, "close"):
                src.close()
        except Exception:
            pass
        try:
            if gen is not None:
                gen.close()
        except Exception:
            pass

    def _open_system(self) -> None:
        self._close_source()
        if self._demo:
            self._source, self._is_demo = SyntheticSource(), True
        else:
            try:
                self._source, self._is_demo = self._system_factory(), False
            except Exception as exc:
                self._source, self._is_demo = SyntheticSource(), True
                self.notice = f"No audio device available ({exc}); showing a demo tone."
        self._rate = self._source.sample_rate
        self._gen = self._source.frames(self._chunk)

    def _bind_app(self) -> bool:
        """Lock onto the wanted app if it is producing audio right now."""
        candidates = [a for a in self._lister.apps if a.key == self._wanted]
        if not candidates:
            return False
        app = max(candidates, key=lambda a: a.playing)
        try:
            source = self._app_factory(app.pid, app.label)
        except Exception as exc:
            label = app.label
            self.selected, self._wanted = SYSTEM, None
            self._open_system()
            self.notice = f"Couldn't capture {label} ({exc}); using all system audio."
            return True
        self._close_source()
        self._source, self._is_demo = source, False
        self._bound_pid = app.pid
        self._rate = source.sample_rate
        self._gen = source.frames(self._chunk)
        self.notice = ""
        return True

    # -- selection ---------------------------------------------------------------

    def select(self, key: str) -> None:
        if key == self.selected:
            return
        self.notice = ""
        if key == SYSTEM or not key.startswith(APP_PREFIX) or not self.supported:
            self.selected = SYSTEM
            self._wanted = None
            self._open_system()
            return
        self.selected = key
        self._wanted = key[len(APP_PREFIX):]
        known = next((a.label for a in self._lister.apps if a.key == self._wanted), self._wanted.capitalize())
        self._wanted_label = known
        self._close_source()                   # silence until the app is bound
        self._lister.request()
        self._lister.wait(0.3)                 # usually already listed; bind straight away when it is
        if not self._bind_app():
            self._last_maintain = self._clock()

    def request_refresh(self) -> None:
        self._lister.request()

    def options(self) -> list[tuple[str, str, str]]:
        """(key, label, note) for every choice, system first, then apps playing first."""
        out = [(SYSTEM, "All system audio", "")]
        seen = set()
        for a in self._lister.apps:
            out.append((APP_PREFIX + a.key, a.label, "playing" if a.playing else "silent"))
            seen.add(a.key)
        if self._wanted and self._wanted not in seen:                  # chosen, but not running right now
            out.append((self.selected, self._wanted_label, "waiting"))
        return out

    @property
    def wanted_key(self) -> str | None:
        """Executable key of the chosen app ('spotify'), or None for all system audio."""
        return self._wanted if self.selected != SYSTEM else None

    @property
    def waiting(self) -> bool:
        return self.selected != SYSTEM and self._gen is None

    @property
    def label(self) -> str:
        if self.selected != SYSTEM:
            return f"Waiting for {self._wanted_label}" if self.waiting else self._source_label()
        return "Demo tone" if self._is_demo else "System audio"

    def _source_label(self) -> str:
        return next((a.label for a in self._lister.apps if a.key == self._wanted), self._wanted_label)

    @property
    def sample_rate(self) -> int:
        src = self._source
        if src is not None:
            self._rate = src.sample_rate
        return self._rate

    # -- per frame ---------------------------------------------------------------

    def _maintain(self) -> None:
        now = self._clock()
        if now - self._last_maintain < self.MAINTAIN_SECONDS:
            return
        self._last_maintain = now
        if self.selected == SYSTEM:
            return
        if self._gen is None:                          # waiting for the app to show up
            self._lister.request()
            self._bind_app()
        elif self._bound_pid is not None and not self._pid_exists(self._bound_pid):
            self._close_source()                       # the app closed: wait for it to come back
            self._lister.request()

    def read(self, n: int) -> np.ndarray:
        """The next chunk of audio (mono or multichannel); silence when there is none."""
        self._chunk = n
        if self._closed:
            return np.zeros(n, dtype=np.float32)
        self._maintain()
        if self._gen is None:
            return np.zeros(n, dtype=np.float32)
        try:
            out = next(self._gen)
            if len(out) != n:                  # a source opened for another size: always hand back exactly n
                out = out[-n:] if len(out) > n else np.concatenate([np.zeros((n - len(out),) + out.shape[1:], out.dtype), out])
            return out
        except StopIteration:
            if self.selected == SYSTEM:
                self._open_system()
            else:
                self._close_source()
        except Exception as exc:
            label = self._wanted_label or "the audio source"
            self.selected, self._wanted = SYSTEM, None
            self._open_system()
            self.notice = f"Audio capture for {label} stopped ({exc}); using all system audio."
        return np.zeros(n, dtype=np.float32)

    def close(self) -> None:
        self._closed = True
        self._close_source()
