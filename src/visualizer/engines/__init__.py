"""Every version of the picture we have built, each as its own engine, and the
fusion layer that runs any combination of them together.

    V1 Spots      the first random-spot field
    V2 Shapes     spinning orbs, spiky orbs, stars, blobs, slashes (+ 3D depth)
    V3 Adaptive   shapes that turn bold when loud and soft when quiet, per-range identity
    V4 Weather    terrain, sand, ribbon, ink, blooms, ripples, grain, hairlines, shards
    V5 Bursts     radial bursts: halos, cores, rings, sunbursts, stars, glow trails
    V6 Ink waves  hollow ink waves spreading across the screen from every hit
    V7 Field      the spectrum as shapes: bass ink, orbs, vocal rings, stars, sand
"""
from .fusion import MIX_KEYS, REGISTRY, VERSIONS, Fusion, default_elements, elements_of, scale_features

__all__ = ["Fusion", "REGISTRY", "VERSIONS", "MIX_KEYS", "default_elements", "elements_of", "scale_features"]
