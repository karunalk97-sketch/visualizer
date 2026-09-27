"""Small image helpers for the scene: fast bilinear resizing (with cached index
tables), smoothstep and seamless value noise."""
from __future__ import annotations

import numpy as np

_resize_cache: dict[tuple[int, int, int, int], tuple] = {}


def _resize_plan(in_h: int, in_w: int, out_h: int, out_w: int) -> tuple:
    key = (in_h, in_w, out_h, out_w)
    if key not in _resize_cache:
        if len(_resize_cache) > 16:
            _resize_cache.clear()
        ys = np.clip((np.arange(out_h, dtype=np.float32) + 0.5) * in_h / out_h - 0.5, 0, in_h - 1)
        xs = np.clip((np.arange(out_w, dtype=np.float32) + 0.5) * in_w / out_w - 0.5, 0, in_w - 1)
        y0 = np.floor(ys).astype(np.intp)
        x0 = np.floor(xs).astype(np.intp)
        _resize_cache[key] = (
            y0, np.minimum(y0 + 1, in_h - 1), x0, np.minimum(x0 + 1, in_w - 1),
            (ys - y0).astype(np.float32)[:, None], (xs - x0).astype(np.float32)[None, :],
        )
    return _resize_cache[key]


def resize_bilinear(arr: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """Smooth resize. Separable (columns first, on the small array)."""
    in_h, in_w = arr.shape
    if (in_h, in_w) == (out_h, out_w):
        return arr.astype(np.float32, copy=True)
    y0, y1, x0, x1, wy, wx = _resize_plan(in_h, in_w, out_h, out_w)
    arr = arr.astype(np.float32, copy=False)
    wide = arr[:, x0] * (1.0 - wx) + arr[:, x1] * wx
    out = wide[y0]
    out *= 1.0 - wy
    out += wide[y1] * wy
    return out


def smoothstep(e0, e1, x):
    """0 below e0, 1 above e1, a smooth S between (e0 > e1 flips it)."""
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def value_noise(lattice: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Seamless (periodic) smooth value noise, 0..1. `lattice` is an (n, n) random
    grid; u, v are coordinates in lattice cells (any shape, broadcastable)."""
    n = lattice.shape[0]
    iu = np.floor(u)
    iv = np.floor(v)
    fu = u - iu
    fv = v - iv
    fu = fu * fu * (3.0 - 2.0 * fu)
    fv = fv * fv * (3.0 - 2.0 * fv)
    i0 = iu.astype(np.intp) % n
    j0 = iv.astype(np.intp) % n
    i1 = (i0 + 1) % n
    j1 = (j0 + 1) % n
    a = lattice[j0, i0]
    b = lattice[j0, i1]
    c = lattice[j1, i0]
    d = lattice[j1, i1]
    top = a + (b - a) * fu
    bot = c + (d - c) * fu
    return (top + (bot - top) * fv).astype(np.float32)
