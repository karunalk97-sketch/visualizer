import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import pytest

import visualizer.main as main_module
from visualizer.main import IDLE_SECONDS, IntensitySmoother, crash_log_path, idle_text, report_crash


def test_a_track_title_is_always_shown_as_is():
    assert idle_text("Song - Band", 10_000, 60) == "Song - Band"
    assert idle_text("Song - Band", 0, 60) == "Song - Band"


# -- IntensitySmoother ("Decay") ---------------------------------------------------

def test_big_screens_stay_within_the_cell_budget_small_ones_keep_their_pixel_size():
    from visualizer.config import Config
    cfg = Config()
    assert main_module.cell_size(cfg, 720, 1280) == cfg.pixel_size            # a normal window: untouched
    for w, h in ((1920, 1080), (2560, 1440), (3840, 2160)):
        gw, gh = main_module.grid_for(cfg, w, h)
        assert gw * gh <= main_module.MAX_GRID_CELLS * 1.02, (w, h)
    cfg.versions = ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]                   # fusing many: smaller budget, bigger cells
    gw, gh = main_module.grid_for(cfg, 1280, 720)
    assert gw * gh <= main_module.MAX_GRID_CELLS / 4 * 1.05
    cfg.render_mode = "chars"
    assert main_module.cell_size(cfg, 2160, 3840) == cfg.glyph_cell           # characters keep their size


def test_zero_decay_passes_each_frame_through_unchanged():
    s = IntensitySmoother()
    a = np.array([[0.0, 1.0]], dtype=np.float32)
    b = np.array([[1.0, 0.0]], dtype=np.float32)
    assert np.array_equal(s.apply(a, 0.0), a)
    assert np.array_equal(s.apply(b, 0.0), b)          # no memory kept between frames


def test_decay_blends_toward_the_new_frame_instead_of_jumping():
    s = IntensitySmoother()
    first = np.zeros((2, 2), dtype=np.float32)
    s.apply(first, 0.5)
    second = np.ones((2, 2), dtype=np.float32)
    blended = s.apply(second, 0.5)
    assert np.allclose(blended, 0.5)                   # halfway between 0 and 1, not snapped to 1
    third = s.apply(second, 0.5)
    assert third.mean() > blended.mean()               # keeps easing toward a held frame, not stuck


def test_higher_decay_lingers_longer():
    low, high = IntensitySmoother(), IntensitySmoother()
    low.apply(np.zeros((2, 2), dtype=np.float32), 0.2)
    high.apply(np.zeros((2, 2), dtype=np.float32), 0.9)
    a = low.apply(np.ones((2, 2), dtype=np.float32), 0.2)
    b = high.apply(np.ones((2, 2), dtype=np.float32), 0.9)
    assert b.mean() < a.mean()                         # high decay: still mostly the old (dark) frame


def test_a_resize_does_not_crash_the_smoother():
    s = IntensitySmoother()
    s.apply(np.zeros((2, 2), dtype=np.float32), 0.5)
    out = s.apply(np.ones((4, 4), dtype=np.float32), 0.5)   # grid size changed underneath it
    assert out.shape == (4, 4)
    assert np.array_equal(out, np.ones((4, 4), dtype=np.float32))  # starts fresh rather than erroring


def test_no_title_and_recent_sound_shows_nothing_special():
    assert idle_text("", 0, 60) == ""
    assert idle_text("", 60 * IDLE_SECONDS - 1, 60) == ""


def test_a_quiet_stretch_with_no_title_explains_itself():
    msg = idle_text("", 60 * IDLE_SECONDS, 60)
    assert "Nothing is playing" in msg


def test_a_crash_is_written_to_a_log_file_with_the_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    try:
        raise ValueError("boom in the render loop")
    except ValueError as exc:
        path = report_crash(exc, show_dialog=False)
    assert path == crash_log_path() == tmp_path / "AudioVisualizer" / "error.log"
    text = path.read_text(encoding="utf-8")
    assert "ValueError: boom in the render loop" in text and "Traceback" in text


def test_crashes_are_appended_not_overwritten(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    for msg in ("first", "second"):
        try:
            raise RuntimeError(msg)
        except RuntimeError as exc:
            report_crash(exc, show_dialog=False)
    text = crash_log_path().read_text(encoding="utf-8")
    assert "first" in text and "second" in text


def test_an_unwritable_log_location_does_not_make_things_worse(monkeypatch, tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    monkeypatch.setenv("LOCALAPPDATA", str(blocker))                   # a file where a folder is needed
    report_crash(RuntimeError("x"), show_dialog=False)                 # must not raise


def test_main_reports_an_unexpected_error_and_exits_cleanly(monkeypatch):
    seen = []
    monkeypatch.setattr(main_module, "run", lambda argv=None: (_ for _ in ()).throw(RuntimeError("kaput")))
    monkeypatch.setattr(main_module, "report_crash", lambda exc, show_dialog=True: seen.append(exc))
    quits = []
    monkeypatch.setattr(main_module.pygame, "quit", lambda: quits.append(1))     # do not shut pygame down for the other tests
    with pytest.raises(SystemExit) as info:
        main_module.main([])
    assert info.value.code == 1 and len(seen) == 1 and str(seen[0]) == "kaput" and quits == [1]


def test_a_normal_exit_or_ctrl_c_is_not_treated_as_a_crash(monkeypatch):
    seen = []
    monkeypatch.setattr(main_module, "report_crash", lambda exc, show_dialog=True: seen.append(exc))
    for exc in (SystemExit(0), KeyboardInterrupt()):
        monkeypatch.setattr(main_module, "run", lambda argv=None, e=exc: (_ for _ in ()).throw(e))
        with pytest.raises(type(exc)):
            main_module.main([])
    assert seen == []
