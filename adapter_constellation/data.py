"""Character-level dataset preparation.

Turns raw text into an integer vocabulary and provides batching of
(input, target) windows for next-character prediction.
"""

from __future__ import annotations

import numpy as np


def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """Builds a deterministic character vocabulary from text.

    Characters are sorted so the mapping is reproducible across runs.
    """
    if not text:
        raise ValueError("cannot build a vocabulary from empty text")
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    """Maps each character of text to its integer id, as an int64 array."""
    try:
        return np.array([stoi[ch] for ch in text], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"character {exc.args[0]!r} is not in the vocabulary") from exc


def decode(ids, itos: dict[int, str]) -> str:
    """Maps a sequence of integer ids back to a string."""
    return "".join(itos[int(i)] for i in ids)


def get_batches(
    data: np.ndarray,
    block_size: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Samples a random batch of (input, target) windows from encoded data.

    `inputs[b]` is a contiguous run of `block_size` ids; `targets[b]` is the
    same run shifted one position to the right, i.e. the next-character
    prediction target at every position.
    """
    if len(data) <= block_size:
        raise ValueError("data must contain more tokens than block_size")
    max_start = len(data) - block_size - 1
    starts = rng.integers(0, max_start + 1, size=batch_size)
    inputs = np.stack([data[s : s + block_size] for s in starts])
    targets = np.stack([data[s + 1 : s + block_size + 1] for s in starts])
    return inputs, targets
