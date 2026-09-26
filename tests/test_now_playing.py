from visualizer.now_playing import NowPlaying, NowPlayingWatcher


def test_label_prefers_title_artist_app():
    assert NowPlaying(app="SPOTIFY", title="SONG", artist="BAND").label() == "SONG - BAND - SPOTIFY"


def test_label_falls_back_to_whatever_is_present():
    assert NowPlaying(app="SPOTIFY").label() == "SPOTIFY"
    assert NowPlaying().label() == "NOW PLAYING: --"


def test_watcher_is_inert_off_windows(monkeypatch):
    monkeypatch.setattr("visualizer.now_playing.sys.platform", "linux")
    watcher = NowPlayingWatcher()
    watcher.start()  # must not spawn a thread or raise on non-Windows platforms
    assert watcher._thread is None
    assert watcher.current() == NowPlaying()
