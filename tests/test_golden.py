"""Golden tests: a fixed seed and a tiny corpus must draw the exact same frame.

Every earlier test checks a *property* of determinism (same inputs give equal
arrays, same model state gives an equal star list). This module pins the
whole pipeline -- dataset prep, base model, LoRA training, delta extraction,
constellation mapping, and text rendering -- down to one literal, byte-for-byte
expected frame. If any step along that chain ever stops being reproducible
(a stray `np.random` call without a seeded generator, a dict iterating in a
different order, a float formatting change), this is the test that catches
it, not just "the two runs disagree with each other."
"""

from __future__ import annotations

import numpy as np

from adapter_constellation.constellation import build_constellation
from adapter_constellation.data import build_vocab, encode, get_batches
from adapter_constellation.lora import lora_sgd_step
from adapter_constellation.lora_model import LoRATransformer
from adapter_constellation.model import TinyTransformer
from adapter_constellation.render import render_frame

# Small enough that training twenty steps and rendering a frame is instant,
# but with enough distinct characters to exercise a real vocabulary.
_CORPUS = "the quick brown fox jumps over the lazy dog"
_BLOCK_SIZE = 8
_STEPS = 20

# Computed once by actually running `_train_and_render` below and reading off
# its output -- this is the "golden" fixture a future change is checked
# against, not a value derived independently of the code it tests.
_GOLDEN_FRAME = [
    "step 20 loss 3.2979     ",
    "        @               ",
    "        +  :@     @     ",
    "    -     @   @  @   @  ",
    "    @  .:@.@ @@         ",
    "    @    @   +  #@   .  ",
    " @ @ @  *         *     ",
    "    .  *   @.     .     ",
    "    .           @:      ",
    "           = .          ",
    "                        ",
]


def _train_and_render(seed: int = 0, steps: int = _STEPS) -> list[str]:
    """Builds a fresh model from `seed`, trains it on the tiny corpus, and renders one frame."""
    stoi, _itos = build_vocab(_CORPUS)
    data = encode(_CORPUS, stoi)
    base = TinyTransformer(
        vocab_size=len(stoi), block_size=_BLOCK_SIZE, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=seed
    )
    lora_model = LoRATransformer(base, rank=3, alpha=2.0, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)

    loss = float("nan")
    for step in range(1, steps + 1):
        inputs, targets = get_batches(data, block_size=_BLOCK_SIZE, batch_size=4, rng=rng)
        loss, grads = lora_model.loss_and_grad(inputs, targets)
        lora_sgd_step(lora_model.parameters(), grads, lr=0.5)

    stars = build_constellation(lora_model, k=3)
    title = f"step {steps} loss {loss:.4f}"
    return render_frame(stars, width=24, height=10, title=title)


def test_same_seed_and_corpus_produce_byte_identical_frames_across_runs():
    # Two completely independent pipelines -- fresh vocab, fresh model, fresh
    # adapters, fresh batch RNG -- from the same seed must agree exactly,
    # line for line and character for character.
    first_run = _train_and_render()
    second_run = _train_and_render()

    assert first_run == second_run
    assert "\n".join(first_run) == "\n".join(second_run)


def test_frame_matches_golden_fixture():
    frame = _train_and_render()

    assert frame == _GOLDEN_FRAME
    for line in frame:
        assert len(line) == 24


def test_a_different_seed_changes_the_frame():
    # Not golden by itself, but guards against the golden test passing only
    # because the pipeline ignores its seed entirely.
    baseline = _train_and_render(seed=0)
    other = _train_and_render(seed=1)

    assert baseline != other
