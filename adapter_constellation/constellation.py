"""Deterministic map from adapter deltas to constellation stars.

The live diagram (a later milestone) just needs a list of stars -- each with
a screen position and a brightness -- to draw. This module is the map that
produces that list from the data `delta.py` already extracts: every
adapter's singular-value spectrum, and (for the attention projections) the
magnitude of its delta restricted to each attention head.

Two properties make the resulting diagram legible rather than noisy:

- A star's *position* depends only on which singular value or head it is
  (its `(layer, target, kind, index)` identity), never on the trained
  numbers. So a star for "layer 0's Wq, head 2" sits in the same spot across
  every training run and every adapter -- the constellation's shape is
  stable, and only how bright each star burns changes as training moves the
  adapter. Position comes from hashing that identity, not from calling
  `random` or `np.random`, so it is reproducible without storing a seed.
- A star's *brightness* depends only on the trained numbers (singular
  values or head-delta norms), normalized against the brightest star in its
  own adapter/target group so brightness is always in `[0, 1]` regardless of
  the adapter's scale.
"""

from __future__ import annotations

import hashlib
import math

import numpy as np

from .delta import extract_deltas, extract_spectra

Array = np.ndarray

# Wq/Wk/Wv produce per-head columns in their *output*; Wo consumes per-head
# rows in its *input* (it mixes heads back together on the way out). W1/W2
# are plain MLP projections with no head structure, so they get no head
# stars at all -- only their singular-value spectrum is drawn.
_HEAD_AXIS = {"Wq": "out", "Wk": "out", "Wv": "out", "Wo": "in"}


def _hash_unit(*parts: str) -> float:
    """Deterministic pseudo-random float in [0, 1) derived from `parts`.

    Uses a cryptographic hash purely as a stable, well-mixed source of bits
    -- there is nothing security-sensitive here, just a reproducible stand-in
    for `random()` that needs no stored seed.
    """
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(1 << 64)


def star_position(layer: int, target: str, kind: str, index: int) -> tuple[float, float]:
    """Deterministic (x, y) in the unit disk for one star's identity.

    `kind` ("singular" or "head") keeps the two star families from ever
    landing on the same point just because they share a layer/target/index.
    Radius is kept away from the exact center (`>= 0.15`) so stars don't
    visually pile up at the origin.
    """
    key = (str(layer), target, kind, str(index))
    angle = _hash_unit("angle", *key) * 2 * math.pi
    radius = 0.15 + 0.85 * _hash_unit("radius", *key)
    return radius * math.cos(angle), radius * math.sin(angle)


def _normalized_brightness(magnitudes: Array) -> Array:
    """Scales `magnitudes` into [0, 1] by its own peak; an all-zero input stays all-zero."""
    peak = float(np.max(magnitudes)) if magnitudes.size else 0.0
    if peak <= 1e-12:
        return np.zeros_like(magnitudes, dtype=float)
    return magnitudes / peak


def head_delta_magnitudes(delta: Array, n_heads: int, axis: str) -> Array:
    """Splits `delta` into `n_heads` equal blocks along `axis` and returns each block's L2 norm.

    `axis="out"` slices columns (for Wq/Wk/Wv, whose output is per-head);
    `axis="in"` slices rows (for Wo, whose input is per-head). Mirrors the
    `split_heads` reshape in `model.py` but needs no reshape here since a
    2-D weight's head columns/rows are already contiguous.
    """
    if axis not in ("in", "out"):
        raise ValueError(f"axis must be 'in' or 'out', got {axis!r}")
    size = delta.shape[0] if axis == "in" else delta.shape[1]
    if size % n_heads != 0:
        raise ValueError(f"dimension {size} is not divisible by n_heads={n_heads}")
    head_size = size // n_heads
    norms = np.empty(n_heads)
    for h in range(n_heads):
        block = delta[h * head_size : (h + 1) * head_size, :] if axis == "in" else delta[:, h * head_size : (h + 1) * head_size]
        norms[h] = np.linalg.norm(block)
    return norms


def spectrum_stars(layer: int, target: str, spectrum: Array) -> list[dict]:
    """One star per singular value of `target`'s delta in layer `layer`."""
    brightness = _normalized_brightness(spectrum)
    stars = []
    for index, (magnitude, glow) in enumerate(zip(spectrum, brightness)):
        x, y = star_position(layer, target, "singular", index)
        stars.append(
            {
                "layer": layer,
                "target": target,
                "kind": "singular",
                "index": index,
                "x": x,
                "y": y,
                "magnitude": float(magnitude),
                "brightness": float(glow),
            }
        )
    return stars


def head_stars(layer: int, target: str, delta: Array, n_heads: int) -> list[dict]:
    """One star per attention head of `target`'s delta, or `[]` if `target` has no head structure."""
    axis = _HEAD_AXIS.get(target)
    if axis is None:
        return []
    magnitudes = head_delta_magnitudes(delta, n_heads, axis)
    brightness = _normalized_brightness(magnitudes)
    stars = []
    for index, (magnitude, glow) in enumerate(zip(magnitudes, brightness)):
        x, y = star_position(layer, target, "head", index)
        stars.append(
            {
                "layer": layer,
                "target": target,
                "kind": "head",
                "index": index,
                "x": x,
                "y": y,
                "magnitude": float(magnitude),
                "brightness": float(glow),
            }
        )
    return stars


def build_constellation(lora_model, k: int | None = None) -> list[dict]:
    """Returns every star -- singular-value and head -- for a `LoRATransformer`.

    `k` optionally truncates each adapter's spectrum (see `delta.delta_svd`)
    before turning it into stars, so a caller can cap how many singular-value
    stars appear per target.
    """
    n_heads = lora_model.base.n_heads
    deltas = extract_deltas(lora_model)
    spectra = extract_spectra(lora_model, k=k)

    stars: list[dict] = []
    for layer_idx, (layer_deltas, layer_spectra) in enumerate(zip(deltas, spectra)):
        for target, spectrum in layer_spectra.items():
            stars.extend(spectrum_stars(layer_idx, target, spectrum))
            stars.extend(head_stars(layer_idx, target, layer_deltas[target], n_heads))
    return stars
