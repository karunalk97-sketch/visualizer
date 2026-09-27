"""Every version, every element, and fusions of them: nothing may crash, leave the
0..1 range, or keep drawing in silence. Plus a few behaviour checks per version."""
import itertools

import numpy as np
import pytest

from visualizer.engines import MIX_KEYS, REGISTRY, VERSIONS, Fusion, default_elements, elements_of, scale_features
from visualizer.engines.base import Inputs
from visualizer.engines.v7_field import AREA_BUDGET, FieldEngine
from visualizer.features import NUM_BANDS, RANGES, Features, band_edges

W, H = 160, 90
DT = 1 / 60
SR = 48000
_, RANGE_OF = band_edges()


def music(i: int, loud: float = 0.8) -> tuple[np.ndarray, Features]:
    """A frame of fake music: a steady chord with a kick every half second and hats between."""
    t = (np.arange(2048) + i * 800) / SR
    kick = 1.0 if i % 30 == 0 else 0.0
    hat = 1.0 if i % 15 == 7 else 0.0
    wave = (loud * (0.3 * np.sin(2 * np.pi * 110 * t) + 0.2 * np.sin(2 * np.pi * 440 * t) * (1 + kick))).astype(np.float32)
    bands = np.clip(loud * (0.5 + 0.3 * np.sin(np.arange(NUM_BANDS) * 0.7 + i * 0.1)), 0, 1).astype(np.float32)
    f = Features(level=loud, low=loud, mid=loud * 0.8, high=loud * 0.6, kick=loud * kick, hit=loud * hat,
                 kick_env=loud * kick, hit_env=loud * hat, brightness=0.5, noisiness=0.3, sustain=loud * 0.7,
                 pitch=0.5, section=0.5, silent=loud == 0, waveform=wave, bands=bands)
    return wave, f


def silence(i: int) -> tuple[np.ndarray, Features]:
    return np.zeros(2048, dtype=np.float32), Features(silent=True)


def run(fusion: Fusion, frames: int, source=music, start: int = 0) -> np.ndarray:
    img = None
    for i in range(start, start + frames):
        wave, f = source(i)
        img = fusion.update(wave, SR, f, DT)
        assert img.shape == (fusion.H, fusion.W) and img.dtype == np.float32
        assert np.isfinite(img).all() and img.min() >= 0.0 and img.max() <= 1.0
    return img


# -- every version on its own -------------------------------------------------

@pytest.mark.parametrize("key", list(REGISTRY))
def test_each_version_draws_music_and_goes_dark_in_silence(key):
    fu = Fusion(W, H, SR, seed=1)
    fu.configure([key])
    img = run(fu, 90)
    assert (img > 0.5).mean() > 0.002, key                     # it shows something for music
    dark = run(fu, 240, silence, start=90)
    assert (dark > 0.5).mean() < 0.002, key                    # and fades out in silence


@pytest.mark.parametrize("key", list(REGISTRY))
def test_every_element_on_its_own_and_all_but_one_never_breaks(key):
    keys = [k for k, _ in elements_of(key)]
    for chosen in [[k] for k in keys] + [[k for k in keys if k != drop] for drop in keys]:
        fu = Fusion(W, H, SR, seed=2)
        fu.configure([key], {key: chosen})
        run(fu, 40)


@pytest.mark.parametrize("key", list(REGISTRY))
def test_versions_survive_resizes_reshuffles_and_new_sections(key):
    fu = Fusion(W, H, SR, seed=3)
    fu.configure([key])
    run(fu, 20)
    fu.resize(120, 70)
    run(fu, 10, start=20)
    fu.reshuffle()
    fu.new_section(0.9)
    fu.resize(W, H)
    run(fu, 10, start=30)


# -- fusion -------------------------------------------------------------------------

def test_every_pair_of_versions_fuses_cleanly():
    for a, b in itertools.combinations(list(REGISTRY), 2):
        fu = Fusion(W, H, SR, seed=4)
        fu.configure([a, b])
        run(fu, 20)


def test_random_fusions_with_random_elements_never_break():
    rng = np.random.default_rng(5)
    keys = list(REGISTRY)
    for _ in range(12):
        versions = list(rng.choice(keys, size=int(rng.integers(1, len(keys) + 1)), replace=False))
        elements = {k: [e for e, _ in elements_of(k) if rng.random() < 0.6] or [elements_of(k)[0][0]] for k in keys}
        fu = Fusion(W, H, SR, seed=int(rng.integers(1000)))
        fu.configure(versions, elements, mix={k: float(rng.uniform(0, 2)) for k in MIX_KEYS},
                     intensity=float(rng.uniform(0, 2)), invert=float(rng.uniform(0.5, 1)))
        run(fu, 25)
        fu.configure(versions[::-1])                              # order changes and versions coming and going
        run(fu, 5, start=25)


def test_all_versions_at_once():
    fu = Fusion(W, H, SR, seed=6)
    fu.configure(list(REGISTRY))
    run(fu, 40)


def test_fused_versions_invert_where_they_overlap():
    a, b = np.full((4, 4), 1.0, np.float32), np.full((4, 4), 1.0, np.float32)
    fu = Fusion(4, 4, SR, seed=7)
    assert np.allclose(a + b - 2.0 * fu.invert * a * b, 0.0)   # white on white turns black


def test_unknown_or_empty_versions_fall_back_to_the_field():
    fu = Fusion(W, H, SR, seed=8)
    fu.configure(["nope"])
    assert fu.versions == ["v7"]
    fu.configure([])
    assert fu.versions == ["v7"]


def test_every_version_is_listed_with_its_elements():
    assert [k for k, _ in VERSIONS] == ["v1", "v2", "v3", "v4", "v5", "v6", "v7"]
    for k, els in default_elements().items():
        assert els and els == [e for e, _ in elements_of(k)]


# -- intensity and mix ------------------------------------------------------------------

def test_intensity_takes_hits_from_subtle_to_jarring():
    _, f = music(0)
    subtle, normal, jarring = (scale_features(f, {}, i) for i in (0.0, 1.0, 2.0))
    assert subtle.kick == 0.0 and subtle.hit_env == 0.0          # subtle: no hits at all
    assert jarring.kick > normal.kick > 0.0
    assert jarring.bands.mean() > normal.bands.mean() > subtle.bands.mean()


def test_mix_turns_each_range_up_or_down():
    _, f = music(0)
    for key, (name, *_ ) in zip(MIX_KEYS, RANGES):
        idx = [r[0] for r in RANGES].index(name)
        off = scale_features(f, {key: 0.0}, 1.0).bands[RANGE_OF == idx]
        up = scale_features(f, {key: 2.0}, 1.0).bands[RANGE_OF == idx]
        assert off.max() == 0.0 and up.mean() > f.bands[RANGE_OF == idx].mean()
    assert scale_features(f, {"bass": 0.0}, 1.0).kick == 0.0


# -- V7 Field specifics -------------------------------------------------------------------

def field_inputs(bands, kick=0.0, hit=0.0):
    f = Features(bands=np.asarray(bands, dtype=np.float32), kick_env=kick, hit_env=hit)
    return Inputs(features=f, band_levels=np.zeros(96, np.float32), bin_gain=np.ones(96, np.float32), intensity=1.0)


def only(range_name, level):
    return np.where(RANGE_OF == [r[0] for r in RANGES].index(range_name), level, 0.0)


def test_field_every_range_draws_its_own_shape():
    for name, *_ in RANGES:
        e = FieldEngine(W, H, seed=3)
        assert (e.update(field_inputs(only(name, 0.8)), DT) > 0.5).mean() > 0.002, name


def test_field_size_follows_loudness_and_quiet_shrinks_away():
    e = FieldEngine(W, H, seed=4)
    lit = lambda v: float((e.update(field_inputs(np.full(NUM_BANDS, v)), DT) > 0.5).mean())  # noqa: E731
    loud, soft, gone = lit(0.9), lit(0.3), lit(0.01)
    assert loud > soft * 1.5 and gone < 0.002


def test_field_never_floods_the_screen():
    e = FieldEngine(W, H, seed=5)
    assert (e.update(field_inputs(np.full(NUM_BANDS, 1.0), kick=1.0, hit=1.0), DT) > 0.5).mean() < AREA_BUDGET + 0.05


def test_field_layout_clumps_rather_than_spreading_evenly():
    e = FieldEngine(W, H, seed=6)
    pts = e._home
    d = np.sqrt(((pts[:, None] - pts[None]) ** 2).sum(-1)) + np.eye(len(pts)) * 9
    nearest = d.min(axis=1)
    even = np.random.default_rng(0).random((len(pts), 2))        # compare with plain uniform randomness
    de = np.sqrt(((even[:, None] - even[None]) ** 2).sum(-1)) + np.eye(len(pts)) * 9
    assert np.median(nearest) < np.median(de.min(axis=1))
