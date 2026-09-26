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
        parts = [p for p in (self.title, self.artist, self.app) if p]
        return " - ".join(parts) if parts else "NOW PLAYING: --"


class NowPlayingWatcher:
    def __init__(self, poll_interval: float = 1.5) -> None:
        self.poll_interval = poll_interval
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

    def _run(self) -> None:
        import asyncio

        async def poll_once() -> NowPlaying:
            from winsdk.windows.media.control import (
                GlobalSystemMediaTransportControlsSessionManager as SessionManager,
            )

            manager = await SessionManager.request_async()
            session = manager.get_current_session()
            if session is None:
                return NowPlaying()

            info = await session.try_get_media_properties_async()
            app_id = session.source_app_user_model_id or ""
            app = app_id.split("!")[0].split(".")[-1] or app_id
            return NowPlaying(
                app=app.upper(),
                title=(info.title or "").upper(),
                artist=(info.artist or "").upper(),
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
