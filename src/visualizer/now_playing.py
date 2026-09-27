"""Reads the current track from Windows' System Media Transport Controls (the
same OS-level API behind the lock-screen media overlay) so the visualizer can
show "what app + what's playing" without any per-app integration (works for
Spotify, browsers, media players, etc. automatically).

The WinRT calls are async and Windows-only, so they run on a background
thread and are polled at a low rate; everything else in the app reads the
latest snapshot via .current() without blocking.
"""
from __future__ import annotations

import sys
import threading
from dataclasses import dataclass


@dataclass
class NowPlaying:
    app: str = ""
    title: str = ""
    artist: str = ""

    def label(self) -> str:
        """'Title - Artist', falling back to whatever is present; '' if nothing is playing."""
        parts = [p for p in (self.title, self.artist) if p]
        if parts:
            return " - ".join(parts)
        return self.app


def clean_app_name(app_id: str) -> str:
    """'Spotify.exe' -> 'Spotify', 'Microsoft.ZuneMusic_8wekyb3d8bbwe!App' -> 'ZuneMusic_8wekyb3d8bbwe'."""
    name = app_id.split("!")[0]
    if name.lower().endswith(".exe"):
        name = name[:-4]
    else:
        name = name.split(".")[-1]
    return name.capitalize() if name.islower() else name


def pick_session_index(app_ids: list[str], preferred: str | None) -> int | None:
    """Index of the media session that belongs to the app we are listening to
    (matched on the executable name inside the session's app id), else None."""
    if not preferred:
        return None
    want = preferred.lower()
    for i, app_id in enumerate(app_ids):
        if want in (app_id or "").lower():
            return i
    return None


class NowPlayingWatcher:
    def __init__(self, poll_interval: float = 1.5) -> None:
        self.poll_interval = poll_interval
        self._preferred: str | None = None
        self._current = NowPlaying()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._supported = sys.platform == "win32"
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self._supported:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def current(self) -> NowPlaying:
        with self._lock:
            return self._current

    def prefer(self, app_key: str | None) -> None:
        """Follow this app's media session (e.g. 'spotify') instead of whichever
        session Windows calls current, so the title matches what we are listening to.
        None goes back to the system's current session."""
        with self._lock:
            if app_key != self._preferred:
                self._preferred = app_key
                self._current = NowPlaying()      # don't show the previous app's track meanwhile

    def _run(self) -> None:
        import asyncio

        async def poll_once() -> NowPlaying:
            try:
                from winsdk.windows.media.control import (
                    GlobalSystemMediaTransportControlsSessionManager as SessionManager,
                )
            except ImportError:  # winsdk has no wheels past Python 3.12; winrt-* is its successor
                from winrt.windows.media.control import (
                    GlobalSystemMediaTransportControlsSessionManager as SessionManager,
                )

            manager = await SessionManager.request_async()
            with self._lock:
                preferred = self._preferred
            sessions = list(manager.get_sessions())
            idx = pick_session_index([s.source_app_user_model_id or "" for s in sessions], preferred)
            if idx is not None:
                session = sessions[idx]
            elif preferred:
                return NowPlaying()               # the app we listen to has no media session: show no (wrong) title
            else:
                session = manager.get_current_session()
            if session is None:
                return NowPlaying()

            info = await session.try_get_media_properties_async()
            app_id = session.source_app_user_model_id or ""
            return NowPlaying(
                app=clean_app_name(app_id) or app_id,
                title=info.title or "",
                artist=info.artist or "",
            )

        loop = asyncio.new_event_loop()
        try:
            while not self._stop.is_set():
                try:
                    result = loop.run_until_complete(poll_once())
                    with self._lock:
                        self._current = result
                except Exception:
                    pass  # no active session, app without SMTC support, etc.
                self._stop.wait(self.poll_interval)
        finally:
            loop.close()
