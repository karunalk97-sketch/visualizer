import numpy as np

from visualizer.audio_apps import AudioApp
from visualizer.audio_router import SYSTEM, AudioRouter

N = 1024


class FakeSource:
    def __init__(self, value=0.1, rate=48000, fail_after=None, end_after=None):
        self.sample_rate = rate
        self.value = value
        self.closed = False
        self._fail_after, self._end_after = fail_after, end_after

    def frames(self, chunk):
        n = 0
        while True:
            if self._end_after is not None and n >= self._end_after:
                return
            if self._fail_after is not None and n >= self._fail_after:
                raise RuntimeError("stream died")
            n += 1
            yield np.full(chunk, self.value, dtype=np.float32)

    def close(self):
        self.closed = True


class FakeLister:
    def __init__(self, apps=()):
        self.apps = list(apps)
        self.requests = 0

    def request(self):
        self.requests += 1

    def wait(self, timeout=0):
        pass


def spotify(pid=10, playing=True):
    return AudioApp("spotify", pid, "Spotify", playing)


def make(apps=(), *, system=None, app_source=None, supported=True, demo=False, alive=lambda pid: True, app_factory=None):
    clock = {"t": 0.0}
    systems = []

    def system_factory():
        if isinstance(system, Exception):
            raise system
        src = system or FakeSource(0.1, 44100)
        systems.append(src)
        return src

    apps_made = []

    def default_app_factory(pid, name):
        src = app_source() if app_source else FakeSource(0.5, 48000)
        src.pid = pid
        apps_made.append(src)
        return src

    lister = FakeLister(apps)
    router = AudioRouter(demo=demo, system_factory=system_factory, app_factory=app_factory or default_app_factory,
                         lister=lister, clock=lambda: clock["t"], pid_exists=alive, supported=supported)
    router.clock, router.lister, router.systems, router.apps_made = clock, lister, systems, apps_made
    return router


def advance(router, seconds=1.5):
    router.clock["t"] += seconds


def test_starts_on_all_system_audio_and_reads_it():
    r = make()
    assert r.selected == SYSTEM and r.label == "System audio" and r.sample_rate == 44100
    assert np.allclose(r.read(N), 0.1)


def test_demo_mode_says_so():
    r = make(demo=True)
    assert r.label == "Demo tone"
    assert len(r.read(N)) == N


def test_no_audio_device_falls_back_to_a_demo_tone_with_an_explanation():
    r = make(system=RuntimeError("no device"))
    assert r.label == "Demo tone" and "no audio device" in r.notice.lower()
    assert len(r.read(N)) == N


def test_selecting_an_app_listens_to_only_that_app():
    r = make([spotify()])
    r.select("app:spotify")
    assert r.selected == "app:spotify" and r.label == "Spotify" and not r.waiting
    assert np.allclose(r.read(N), 0.5) and r.sample_rate == 48000        # the app's audio, not the system's
    assert r.systems[0].closed                                            # the system capture was released
    assert r.apps_made and r.apps_made[0].pid == 10


def test_going_back_to_system_audio_releases_the_app():
    r = make([spotify()])
    r.select("app:spotify")
    app = r.apps_made[0]
    r.select(SYSTEM)
    assert app.closed and r.label == "System audio" and np.allclose(r.read(N), 0.1)


def test_choosing_an_app_that_is_not_running_waits_then_locks_on_when_it_appears():
    r = make([])
    r.select("app:spotify")
    assert r.waiting and r.label == "Waiting for Spotify"
    assert np.allclose(r.read(N), 0.0) and len(r.read(N)) == N            # silence, not an error
    r.lister.apps = [spotify()]                                           # it starts playing
    advance(r)
    assert np.allclose(r.read(N), 0.5) and not r.waiting and r.label == "Spotify"


def test_when_the_app_closes_it_waits_and_rebinds_when_it_comes_back():
    alive = {"ok": True}
    r = make([spotify(10)], alive=lambda pid: alive["ok"])
    r.select("app:spotify")
    assert np.allclose(r.read(N), 0.5)
    alive["ok"] = False
    r.lister.apps = []
    advance(r)
    r.read(N)
    assert r.waiting
    alive["ok"] = True
    r.lister.apps = [spotify(22)]
    advance(r)
    assert np.allclose(r.read(N), 0.5) and r.apps_made[-1].pid == 22      # the new process id


def test_a_capture_that_cannot_start_falls_back_to_system_audio_and_says_why():
    def refuse(pid, name):
        raise OSError("access denied")
    r = make([spotify()], app_factory=refuse)
    r.select("app:spotify")
    assert r.selected == SYSTEM and r.label == "System audio"
    assert "Spotify" in r.notice and "access denied" in r.notice
    assert np.allclose(r.read(N), 0.1)


def test_a_capture_that_dies_mid_stream_falls_back_instead_of_crashing():
    r = make([spotify()], app_source=lambda: FakeSource(0.5, fail_after=3))
    r.select("app:spotify")
    for _ in range(3):
        assert np.allclose(r.read(N), 0.5)
    out = r.read(N)                                                       # the stream raises here
    assert len(out) == N and r.selected == SYSTEM and "stopped" in r.notice
    assert np.allclose(r.read(N), 0.1)


def test_a_system_stream_that_ends_is_reopened_not_fatal():
    sources = [FakeSource(0.1, end_after=2), FakeSource(0.3)]
    it = iter(sources)
    clock = {"t": 0.0}
    r = AudioRouter(system_factory=lambda: next(it), lister=FakeLister(), clock=lambda: clock["t"], supported=True)
    for _ in range(2):
        assert np.allclose(r.read(N), 0.1)
    assert len(r.read(N)) == N                                            # ended: silence for a moment, no StopIteration
    assert np.allclose(r.read(N), 0.3)                                    # and it is back on a fresh stream


def test_read_never_raises_and_always_returns_the_requested_length():
    r = make([])
    r.select("app:nothing")
    for n in (1, 480, 1024, 4096):
        assert r.read(n).shape == (n,)


def test_unsupported_machines_stay_on_system_audio():
    r = make([spotify()], supported=False)
    r.select("app:spotify")
    assert r.selected == SYSTEM and r.label == "System audio"


def test_selecting_the_same_source_again_changes_nothing():
    r = make([spotify()])
    r.select("app:spotify")
    made = len(r.apps_made)
    r.select("app:spotify")
    assert len(r.apps_made) == made


def test_options_list_system_first_then_apps_and_the_chosen_app_even_if_gone():
    r = make([spotify(), AudioApp("steam", 5, "Steam", False)])
    assert r.options() == [(SYSTEM, "All system audio", ""), ("app:spotify", "Spotify", "playing"), ("app:steam", "Steam", "silent")]
    r.select("app:spotify")
    r.lister.apps = []                                                    # it closed
    assert ("app:spotify", "Spotify", "waiting") in r.options()


def test_the_sample_rate_follows_the_active_source():
    r = make([spotify()])
    assert r.sample_rate == 44100
    r.select("app:spotify")
    assert r.sample_rate == 48000
    r.select(SYSTEM)
    assert r.sample_rate == 44100


def test_closing_releases_everything():
    r = make([spotify()])
    r.select("app:spotify")
    r.close()
    assert r.apps_made[0].closed
    assert np.allclose(r.read(N), 0.0)
