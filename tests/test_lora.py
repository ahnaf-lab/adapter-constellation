import numpy as np

from adapter_constellation.lora import effective_weight, init_adapter, lora_sgd_step
from adapter_constellation.lora_model import DEFAULT_TARGETS, LoRATransformer
from adapter_constellation.model import TinyTransformer


def make_base(seed=0):
    return TinyTransformer(
        vocab_size=6, block_size=5, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=seed
    )


def test_adapter_is_a_no_op_at_init():
    rng = np.random.default_rng(0)
    adapter = init_adapter(rng, in_dim=4, out_dim=4, rank=2, alpha=1.0)
    W = rng.normal(size=(4, 4))
    assert np.array_equal(effective_weight(W, adapter), W)


def test_forward_matches_base_before_training():
    base = make_base()
    lora_model = LoRATransformer(base, rank=2, seed=1)
    idx = np.array([[0, 1, 2, 3]])

    base_logits = base.forward(idx)
    lora_logits = lora_model.forward(idx)
    assert np.allclose(base_logits, lora_logits)


def test_base_weights_are_never_modified_by_training():
    base = make_base(seed=2)
    original = {name: base.params["layers"][0][name].copy() for name in DEFAULT_TARGETS}
    lora_model = LoRATransformer(base, rank=2, seed=3)

    rng = np.random.default_rng(4)
    idx = rng.integers(0, base.vocab_size, size=(3, base.block_size - 1))
    targets = rng.integers(0, base.vocab_size, size=(3, base.block_size - 1))

    for _ in range(5):
        loss, grads = lora_model.loss_and_grad(idx, targets)
        lora_sgd_step(lora_model.parameters(), grads, lr=0.5)

    for name in DEFAULT_TARGETS:
        assert np.array_equal(base.params["layers"][0][name], original[name])


def test_training_reduces_loss():
    base = make_base(seed=5)
    lora_model = LoRATransformer(base, rank=4, alpha=2.0, seed=6)

    rng = np.random.default_rng(7)
    idx = rng.integers(0, base.vocab_size, size=(4, base.block_size - 1))
    targets = rng.integers(0, base.vocab_size, size=(4, base.block_size - 1))

    first_loss, grads = lora_model.loss_and_grad(idx, targets)
    lora_sgd_step(lora_model.parameters(), grads, lr=0.5)
    for _ in range(50):
        loss, grads = lora_model.loss_and_grad(idx, targets)
        lora_sgd_step(lora_model.parameters(), grads, lr=0.5)
    final_loss, _ = lora_model.loss_and_grad(idx, targets)

    assert final_loss < first_loss


def test_adapter_shapes_match_targeted_weights():
    base = make_base(seed=8)
    lora_model = LoRATransformer(base, rank=3, targets=("Wq", "W2"), seed=9)

    for layer_idx, layer_params in enumerate(base.params["layers"]):
        for name in ("Wq", "W2"):
            in_dim, out_dim = layer_params[name].shape
            adapter = lora_model.adapters[layer_idx][name]
            assert adapter["A"].shape == (in_dim, 3)
            assert adapter["B"].shape == (3, out_dim)


def test_gradient_matches_finite_difference():
    base = make_base(seed=10)
    lora_model = LoRATransformer(base, rank=2, seed=11)
    rng = np.random.default_rng(12)
    idx = rng.integers(0, base.vocab_size, size=(2, base.block_size - 1))
    targets = rng.integers(0, base.vocab_size, size=(2, base.block_size - 1))

    # Train a few steps first so B moves away from its zero initialization;
    # at init, dA is exactly zero (it scales with B), which would make the
    # check below trivially pass without actually exercising the gradient.
    for _ in range(3):
        loss, grads = lora_model.loss_and_grad(idx, targets)
        lora_sgd_step(lora_model.parameters(), grads, lr=0.5)

    loss, grads = lora_model.loss_and_grad(idx, targets)
    checks = [
        (lora_model.adapters[0]["Wq"]["A"], grads["layers"][0]["Wq"]["A"]),
        (lora_model.adapters[0]["Wq"]["B"], grads["layers"][0]["Wq"]["B"]),
        (lora_model.adapters[1]["W2"]["B"], grads["layers"][1]["W2"]["B"]),
    ]
    h = 1e-4
    for param_array, grad_array in checks:
        flat_param = param_array.reshape(-1)
        flat_grad = grad_array.reshape(-1)
        positions = rng.choice(flat_param.size, size=min(3, flat_param.size), replace=False)
        for pos in positions:
            original = flat_param[pos]

            flat_param[pos] = original + h
            loss_plus, _ = lora_model.loss_and_grad(idx, targets)

            flat_param[pos] = original - h
            loss_minus, _ = lora_model.loss_and_grad(idx, targets)

            flat_param[pos] = original
            numerical = (loss_plus - loss_minus) / (2 * h)
            analytical = flat_grad[pos]
            assert abs(numerical - analytical) < 1e-3 + 1e-2 * abs(numerical), (
                f"gradient mismatch at pos {pos}: numerical={numerical}, analytical={analytical}"
            )
