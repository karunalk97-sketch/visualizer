import sys

from visualizer import audio_apps
from visualizer.audio_apps import AppLister, AudioApp, collect_apps, label_for


def test_friendly_labels():
    assert label_for("Spotify.exe") == "Spotify"
    assert label_for("msedge.exe") == "Edge"
    assert label_for("chrome.exe") == "Chrome"
    assert label_for("someplayer.exe") == "Someplayer"      # unknown, all lower-case: capitalised
    assert label_for("MyPlayer.exe") == "MyPlayer"          # unknown, mixed case: left alone


def test_sessions_are_grouped_by_executable_and_the_playing_one_wins():
    apps = collect_apps([(100, "chrome.exe", 0), (101, "chrome.exe", 1), (102, "chrome.exe", 0)], own_pid=1)
    assert len(apps) == 1
    assert apps[0] == AudioApp(key="chrome", pid=101, label="Chrome", playing=True)


def test_playing_apps_come_first_then_alphabetical():
    apps = collect_apps([(1, "zeta.exe", 0), (2, "alpha.exe", 0), (3, "Spotify.exe", 1), (4, "vlc.exe", 1)], own_pid=99)
    assert [a.key for a in apps] == ["spotify", "vlc", "alpha", "zeta"]
    assert [a.playing for a in apps] == [True, True, False, False]


def test_system_plumbing_expired_sessions_and_ourselves_are_left_out():
    apps = collect_apps([
        (0, "System Sounds", 1),            # no process
        (5, "audiodg.exe", 1),              # windows audio engine
        (6, "AudioVisualizer.exe", 1),      # this app
        (7, "python.exe", 1),
        (8, "gone.exe", 2),                 # expired session
        (9, "", 1),                         # no name
        (10, "own.exe", 1),                 # our own pid
        (11, "Spotify.exe", 1),
    ], own_pid=10)
    assert [a.key for a in apps] == ["spotify"]


def test_no_sessions_gives_an_empty_list():
    assert collect_apps([], own_pid=1) == []


def test_lister_refreshes_in_the_background_and_hands_out_copies():
    calls = []
    lister = AppLister(fetch=lambda: calls.append(1) or [AudioApp("x", 1, "X", True)], min_interval=0.0)
    assert lister.apps == []
    lister.request()
    lister.wait()
    assert lister.apps == [AudioApp("x", 1, "X", True)] and len(calls) == 1
    lister.apps.clear()                                   # a copy: cannot corrupt the cache
    assert len(lister.apps) == 1


def test_lister_does_not_refresh_more_often_than_asked():
    now = [0.0]
    calls = []
    lister = AppLister(fetch=lambda: calls.append(1) or [], min_interval=1.0, clock=lambda: now[0])
    for _ in range(3):
        lister.request(); lister.wait()
    assert len(calls) == 1
    now[0] = 1.5
    lister.request(); lister.wait()
    assert len(calls) == 2


def test_a_failing_fetch_never_breaks_the_lister():
    def boom():
        raise RuntimeError("com error")
    lister = AppLister(fetch=boom, min_interval=0.0)
    lister.request(); lister.wait()
    assert lister.apps == []
    lister.request(); lister.wait()                       # the lock was released: it can try again


def test_per_app_capture_is_reported_unsupported_off_windows(monkeypatch):
    monkeypatch.setattr(sys, "platform", "darwin")
    assert audio_apps.per_app_supported() is False
    assert audio_apps.list_audio_apps() == []
