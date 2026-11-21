"""Delta extraction: SVD of each LoRA adapter's effective weight update.

A LoRA adapter's actual contribution to a weight matrix is
`delta = scale * A @ B`. Because `A` is `(in_dim, rank)` and `B` is
`(rank, out_dim)`, `delta` has rank at most the adapter's rank no matter how
large the underlying weight is -- so its singular-value spectrum (and the
directions that go with the top few values) is a compact summary of what the
adapter learned. This module computes that spectrum for one adapter or for
every adapter in a `LoRATransformer`, and exposes a small tracker a training
loop can use to do so every N steps rather than every step. The later
constellation visualization is drawn from exactly this data.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def adapter_delta(adapter: dict) -> Array:
    """Returns `scale * A @ B`, the adapter's raw contribution to its weight.

    Unlike `lora.effective_weight`, this does not need (or add) the frozen
    base weight `W` -- it is exactly the matrix SVD should decompose.
    """
    return adapter["scale"] * (adapter["A"] @ adapter["B"])


def delta_svd(delta: Array, k: int | None = None) -> dict:
    """Singular value decomposition of a delta matrix.

    A thin wrapper around `numpy.linalg.svd` (`full_matrices=False`),
    returning the left/right singular vectors and singular values, optionally
    truncated to the top `k`. Since `delta` is rank <= adapter rank by
    construction, singular values past that rank are numerically ~0.
    """
    U, S, Vt = np.linalg.svd(delta, full_matrices=False)
    if k is not None:
        U, S, Vt = U[:, :k], S[:k], Vt[:k, :]
    return {"U": U, "S": S, "Vt": Vt}


def adapter_spectrum(adapter: dict, k: int | None = None) -> Array:
    """Convenience: the singular values alone of an adapter's delta."""
    return delta_svd(adapter_delta(adapter), k=k)["S"]


def extract_deltas(lora_model) -> list[dict]:
    """Returns every layer's/target's delta matrix from a `LoRATransformer`.

    Shape mirrors `lora_model.adapters`: a list (one entry per layer) of
    dicts (one entry per target name) of delta matrices.
    """
    return [
        {name: adapter_delta(adapter) for name, adapter in layer_adapters.items()}
        for layer_adapters in lora_model.adapters
    ]


def extract_spectra(lora_model, k: int | None = None) -> list[dict]:
    """Returns every layer's/target's singular-value spectrum from a `LoRATransformer`."""
    return [
        {name: adapter_spectrum(adapter, k=k) for name, adapter in layer_adapters.items()}
        for layer_adapters in lora_model.adapters
    ]


def is_due(step: int, every_n: int) -> bool:
    """Whether a completed training `step` should trigger a spectrum extraction.

    Steps are counted from 1 upward; `every_n=5` fires on steps 5, 10, 15,
    ... and never on step 0, so a tracker never records before any training
    has actually happened.
    """
    if every_n < 1:
        raise ValueError("every_n must be >= 1")
    return step > 0 and step % every_n == 0


class SpectrumTracker:
    """Records adapter spectra from a `LoRATransformer` every N training steps.

    A training loop calls `maybe_record(step)` once per completed step; it
    appends a `{"step": step, "spectra": ...}` snapshot to `history` whenever
    `step` is due per `is_due`, and does nothing (returning `None`) otherwise.
    This is the hook the later live constellation display reads from, so SVDs
    are only ever computed on the steps that will actually be shown.
    """

    def __init__(self, lora_model, every_n: int, k: int | None = None):
        self.lora_model = lora_model
        self.every_n = every_n
        self.k = k
        self.history: list[dict] = []

    def maybe_record(self, step: int) -> dict | None:
        if not is_due(step, self.every_n):
            return None
        snapshot = {"step": step, "spectra": extract_spectra(self.lora_model, k=self.k)}
        self.history.append(snapshot)
        return snapshot
