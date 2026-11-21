import math

import numpy as np

from adapter_constellation.constellation import (
    build_constellation,
    head_delta_magnitudes,
    spectrum_stars,
    star_position,
)
from adapter_constellation.lora import init_adapter, lora_sgd_step
from adapter_constellation.lora_model import LoRATransformer
from adapter_constellation.model import TinyTransformer


def make_base(seed=0):
    return TinyTransformer(
        vocab_size=6, block_size=5, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=seed
    )


def make_trained_lora(seed=0, rank=3, steps=10):
    base = make_base(seed=seed)
    lora_model = LoRATransformer(base, rank=rank, alpha=2.0, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    idx = rng.integers(0, base.vocab_size, size=(3, base.block_size - 1))
    targets = rng.integers(0, base.vocab_size, size=(3, base.block_size - 1))
    for _ in range(steps):
        loss, grads = lora_model.loss_and_grad(idx, targets)
        lora_sgd_step(lora_model.parameters(), grads, lr=0.5)
    return lora_model


def test_star_position_is_deterministic_across_calls():
    first = star_position(1, "Wq", "singular", 2)
    second = star_position(1, "Wq", "singular", 2)
    assert first == second


def test_star_position_depends_on_identity_not_on_magnitude():
    # Two adapters trained differently have different singular values but,
    # for the same (layer, target, kind, index), must land on the same spot.
    lora_a = make_trained_lora(seed=0)
    lora_b = make_trained_lora(seed=100)

    stars_a = {(s["target"], s["kind"], s["index"]): s for s in spectrum_stars(0, "Wq", np.array([1.0, 2.0, 3.0]))}
    stars_b = {(s["target"], s["kind"], s["index"]): s for s in spectrum_stars(0, "Wq", np.array([9.0, 8.0, 7.0]))}

    for key in stars_a:
        assert (stars_a[key]["x"], stars_a[key]["y"]) == (stars_b[key]["x"], stars_b[key]["y"])
        # But the magnitudes differ, so brightness should not be forced equal.
    assert stars_a[("Wq", "singular", 0)]["brightness"] != stars_b[("Wq", "singular", 0)]["brightness"]
    # Sanity: lora_a/lora_b were only built to prove the two seeds genuinely differ.
    assert not np.array_equal(lora_a.adapters[0]["Wq"]["B"], lora_b.adapters[0]["Wq"]["B"])


def test_star_position_lands_in_unit_disk():
    for index in range(5):
        x, y = star_position(3, "W2", "singular", index)
        radius = math.hypot(x, y)
        assert 0.15 <= radius < 1.0


def test_spectrum_stars_brightness_ranks_with_magnitude():
    spectrum = np.array([5.0, 2.0, 1.0])
    stars = spectrum_stars(0, "Wq", spectrum)

    assert [s["index"] for s in stars] == [0, 1, 2]
    assert stars[0]["brightness"] == 1.0
    assert stars[0]["brightness"] > stars[1]["brightness"] > stars[2]["brightness"]
    for s in stars:
        assert 0.0 <= s["brightness"] <= 1.0


def test_spectrum_stars_all_zero_magnitude_gives_zero_brightness():
    stars = spectrum_stars(0, "W1", np.zeros(3))
    assert all(s["brightness"] == 0.0 for s in stars)


def test_head_delta_magnitudes_matches_manual_block_norms():
    delta = np.array(
        [
            [1.0, 0.0, 3.0, 0.0],
            [0.0, 1.0, 0.0, 4.0],
        ]
    )
    norms = head_delta_magnitudes(delta, n_heads=2, axis="out")
    assert norms.shape == (2,)
    assert math.isclose(norms[0], math.sqrt(1.0**2 + 1.0**2))
    assert math.isclose(norms[1], math.sqrt(3.0**2 + 4.0**2))


def test_head_delta_magnitudes_rejects_indivisible_dimension():
    delta = np.zeros((2, 5))
    try:
        head_delta_magnitudes(delta, n_heads=2, axis="out")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a dimension not divisible by n_heads")


def test_build_constellation_covers_every_target_with_singular_and_head_stars():
    lora_model = make_trained_lora(rank=3)
    stars = build_constellation(lora_model, k=3)

    by_target_kind = {}
    for star in stars:
        by_target_kind.setdefault((star["layer"], star["target"], star["kind"]), []).append(star)

    # 2 layers x 6 targets x singular stars (truncated to k=3 each), plus head
    # stars only for the four attention projections (n_heads=2 each).
    singular_groups = [k for k in by_target_kind if k[2] == "singular"]
    head_groups = [k for k in by_target_kind if k[2] == "head"]
    assert len(singular_groups) == 2 * 6
    assert len(head_groups) == 2 * 4
    for key in singular_groups:
        assert len(by_target_kind[key]) == 3
    for key in head_groups:
        assert len(by_target_kind[key]) == 2


def test_build_constellation_is_deterministic_for_the_same_model_state():
    lora_model = make_trained_lora()
    first = build_constellation(lora_model)
    second = build_constellation(lora_model)
    assert first == second
