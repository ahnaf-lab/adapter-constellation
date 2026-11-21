import numpy as np

from adapter_constellation.live import (
    DEFAULT_BLOCK_SIZE,
    build_default_setup,
    draw_frame,
    run_live_training,
)


class FakeScreen:
    """A recording stand-in for a curses window, exposing only erase/addstr/refresh."""

    def __init__(self):
        self.frames: list[list[str]] = []
        self._current: dict[int, str] = {}
        self.refresh_calls = 0

    def erase(self):
        self._current = {}

    def addstr(self, row, col, text):
        assert col == 0
        self._current[row] = text

    def refresh(self):
        self.refresh_calls += 1
        ordered = [self._current[r] for r in sorted(self._current)]
        self.frames.append(ordered)


def test_draw_frame_writes_every_line_and_refreshes_once():
    screen = FakeScreen()
    draw_frame(screen, ["abc", "def"])
    assert screen.frames == [["abc", "def"]]
    assert screen.refresh_calls == 1


def test_build_default_setup_is_deterministic():
    lora_a, data_a, _rng_a = build_default_setup()
    lora_b, data_b, _rng_b = build_default_setup()

    assert np.array_equal(data_a, data_b)
    assert np.array_equal(lora_a.adapters[0]["Wq"]["A"], lora_b.adapters[0]["Wq"]["A"])
    assert np.array_equal(lora_a.base.params["wte"], lora_b.base.params["wte"])


def test_run_live_training_draws_a_frame_exactly_every_every_n_steps():
    lora_model, data, rng = build_default_setup()
    screen = FakeScreen()

    run_live_training(
        screen,
        lora_model,
        data,
        rng,
        steps=12,
        every_n=4,
        lr=0.5,
        batch_size=2,
        block_size=DEFAULT_BLOCK_SIZE,
        k=2,
        width=20,
        height=6,
    )

    assert screen.refresh_calls == 3  # steps 4, 8, 12
    for frame in screen.frames:
        assert len(frame) == 7  # title line + height rows


def test_run_live_training_reduces_loss_over_many_steps():
    lora_model, data, rng = build_default_setup()
    screen = FakeScreen()

    final_loss = run_live_training(
        screen,
        lora_model,
        data,
        rng,
        steps=1,
        every_n=1,
        lr=0.5,
        batch_size=2,
        block_size=DEFAULT_BLOCK_SIZE,
        k=2,
    )
    first_loss = final_loss

    last_loss = run_live_training(
        screen,
        lora_model,
        data,
        rng,
        steps=200,
        every_n=1000,
        lr=0.5,
        batch_size=4,
        block_size=DEFAULT_BLOCK_SIZE,
        k=2,
    )

    assert last_loss < first_loss
