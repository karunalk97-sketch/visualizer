from visualizer.now_playing import NowPlaying, NowPlayingWatcher, clean_app_name


def test_label_is_title_and_artist_in_normal_case():
    assert NowPlaying(app="Spotify", title="Song", artist="Band").label() == "Song - Band"


def test_label_falls_back_to_whatever_is_present():
    assert NowPlaying(title="Song").label() == "Song"
    assert NowPlaying(app="Spotify").label() == "Spotify"
    assert NowPlaying().label() == ""


def test_clean_app_name():
    assert clean_app_name("Spotify.exe") == "Spotify"
    assert clean_app_name("chrome.exe") == "Chrome"
    assert clean_app_name("Microsoft.ZuneMusic_8wekyb3d8bbwe!Microsoft.ZuneMusic") == "ZuneMusic_8wekyb3d8bbwe"


def test_watcher_is_inert_off_windows(monkeypatch):
    monkeypatch.setattr("visualizer.now_playing.sys.platform", "linux")
    watcher = NowPlayingWatcher()
    watcher.start()  # must not spawn a thread or raise on non-Windows platforms
    assert watcher._thread is None
    assert watcher.current() == NowPlaying()
