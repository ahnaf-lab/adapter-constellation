"""Live display: redraws the constellation on screen as LoRA training runs.

The training loop itself is nothing new -- it is the same
`loss_and_grad`/`lora_sgd_step` pair earlier milestones already exercise.
What this module adds is drawing: every `every_n` steps it turns the
adapters' current state into stars (`constellation.build_constellation`),
renders them to text (`render.render_frame`), and writes that text to a
screen.

The screen is kept behind a tiny interface -- `erase()`, `addstr(row, col,
text)`, `refresh()` -- which is exactly the subset of the standard `curses`
window API this module needs. `run_live_training` only ever calls through
that interface, so it never cares whether it was handed a real curses
window or a plain in-memory stand-in; `main()` is the only place a real
terminal (`curses.wrapper`) enters the picture. Tests drive
`run_live_training` with a recording fake and never open a real terminal.
"""

from __future__ import annotations

import curses

import numpy as np

from .constellation import build_constellation
from .data import build_vocab, encode, get_batches
from .delta import is_due
from .lora import lora_sgd_step
from .lora_model import LoRATransformer
from .model import TinyTransformer
from .render import render_frame

# A fixed seed and corpus so a run of this module always trains on the same
# data in the same order -- the live diagram's shape is reproducible, and
# only run-to-run timing (not content) differs.
DEFAULT_SEED = 0
DEFAULT_CORPUS = (
    "the quick brown fox jumps over the lazy dog while the slow turtle "
    "watches from beneath the old oak tree and the river runs quietly by "
    "until the first star of the evening finally appears above the hill"
)
DEFAULT_BLOCK_SIZE = 16
DEFAULT_WIDTH = 60
DEFAULT_HEIGHT = 20


def draw_frame(screen, lines: list[str]) -> None:
    """Draws one rendered frame (a list of text lines) onto `screen`.

    Lines that do not fit the real terminal's current size raise
    `curses.error` from `addstr`; those are skipped so a window resize or a
    too-wide frame cannot crash a training run mid-step.
    """
    screen.erase()
    for row, line in enumerate(lines):
        try:
            screen.addstr(row, 0, line)
        except curses.error:
            continue
    screen.refresh()


def build_default_setup(seed: int = DEFAULT_SEED, corpus: str = DEFAULT_CORPUS):
    """Builds the fixed base model, LoRA wrapper, encoded corpus, and batch RNG.

    Deterministic in `seed` and `corpus`: calling this twice with the same
    arguments produces bit-identical models and data, which is what lets the
    live display be rerun and still draw the same constellation shape.
    """
    stoi, _itos = build_vocab(corpus)
    data = encode(corpus, stoi)
    base = TinyTransformer(vocab_size=len(stoi), block_size=DEFAULT_BLOCK_SIZE, seed=seed)
    lora_model = LoRATransformer(base, rank=4, alpha=2.0, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    return lora_model, data, rng


def run_live_training(
    screen,
    lora_model,
    data: np.ndarray,
    rng: np.random.Generator,
    steps: int,
    every_n: int,
    lr: float,
    batch_size: int = 8,
    block_size: int = DEFAULT_BLOCK_SIZE,
    k: int | None = None,
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
) -> float:
    """Runs `steps` LoRA training steps, redrawing the constellation every `every_n` steps.

    Returns the final step's loss. `screen` only needs to satisfy the
    `erase`/`addstr`/`refresh` interface `draw_frame` uses.
    """
    loss = float("nan")
    for step in range(1, steps + 1):
        inputs, targets = get_batches(data, block_size=block_size, batch_size=batch_size, rng=rng)
        loss, grads = lora_model.loss_and_grad(inputs, targets)
        lora_sgd_step(lora_model.parameters(), grads, lr=lr)
        if is_due(step, every_n):
            stars = build_constellation(lora_model, k=k)
            title = f"step {step}/{steps}  loss {loss:.4f}"
            draw_frame(screen, render_frame(stars, width, height, title=title))
    return loss


def main(steps: int = 300, every_n: int = 5, lr: float = 0.5, batch_size: int = 8, k: int | None = 4) -> None:
    """Entry point: trains on the fixed default corpus, drawing live to a real terminal."""
    lora_model, data, rng = build_default_setup()

    def _run(stdscr):
        curses.curs_set(0)
        run_live_training(stdscr, lora_model, data, rng, steps, every_n, lr, batch_size=batch_size, k=k)

    curses.wrapper(_run)


if __name__ == "__main__":
    main()
