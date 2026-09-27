from visualizer.now_playing import NowPlaying, NowPlayingWatcher, clean_app_name, pick_session_index


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


def test_the_track_comes_from_the_session_of_the_app_we_listen_to():
    ids = ["Chrome", "Spotify.exe", "MSEdge"]
    assert pick_session_index(ids, "spotify") == 1
    assert pick_session_index(ids, "MSEDGE") == 2                  # case does not matter
    assert pick_session_index(ids, "chrome") == 0


def test_no_matching_session_or_no_preference_gives_none():
    assert pick_session_index(["Spotify.exe"], "vlc") is None
    assert pick_session_index(["Spotify.exe"], None) is None
    assert pick_session_index(["Spotify.exe"], "") is None
    assert pick_session_index([], "spotify") is None
    assert pick_session_index(["", None], "spotify") is None


def test_changing_the_preferred_app_clears_the_old_title_right_away():
    w = NowPlayingWatcher()
    w._current = NowPlaying(app="Spotify", title="Song", artist="Band")
    w.prefer("spotify")
    assert w.current() == NowPlaying()                              # no stale title while the new session is looked up
    w._current = NowPlaying(title="Song")
    w.prefer("spotify")                                             # unchanged: keep what we have
    assert w.current().title == "Song"
    w.prefer(None)
    assert w.current() == NowPlaying()