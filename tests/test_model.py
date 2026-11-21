import numpy as np

from adapter_constellation.model import TinyTransformer, cross_entropy_loss, sgd_step, softmax


def make_model(seed=0):
    return TinyTransformer(
        vocab_size=6, block_size=5, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=seed
    )


def _assert_grad_close(model, idx, targets, param_array, grad_array, rng, n_checks=4, h=1e-4):
    flat_param = param_array.reshape(-1)
    flat_grad = grad_array.reshape(-1)
    positions = rng.choice(flat_param.size, size=min(n_checks, flat_param.size), replace=False)
    for pos in positions:
        original = flat_param[pos]

        flat_param[pos] = original + h
        loss_plus, _ = model.loss_and_grad(idx, targets)

        flat_param[pos] = original - h
        loss_minus, _ = model.loss_and_grad(idx, targets)

        flat_param[pos] = original
        numerical = (loss_plus - loss_minus) / (2 * h)
        analytical = flat_grad[pos]
        assert abs(numerical - analytical) < 1e-3 + 1e-2 * abs(numerical), (
            f"gradient mismatch at pos {pos}: numerical={numerical}, analytical={analytical}"
        )


def test_forward_shape():
    model = make_model()
    idx = np.array([[0, 1, 2, 3], [2, 3, 4, 5]])
    logits = model.forward(idx)
    assert logits.shape == (2, 4, 6)


def test_loss_and_grad_shapes_match_params():
    model = make_model()
    idx = np.array([[0, 1, 2, 3]])
    targets = np.array([[1, 2, 3, 4]])
    loss, grads = model.loss_and_grad(idx, targets)
    assert np.isscalar(loss) or loss.shape == ()
    assert grads["wte"].shape == model.params["wte"].shape
    assert grads["head_W"].shape == model.params["head_W"].shape
    assert len(grads["layers"]) == model.n_layers


def test_gradient_matches_finite_difference():
    model = make_model(seed=1)
    rng = np.random.default_rng(42)
    idx = rng.integers(0, model.vocab_size, size=(2, 4))
    targets = rng.integers(0, model.vocab_size, size=(2, 4))
    loss, grads = model.loss_and_grad(idx, targets)

    checks = [
        (model.params["layers"][0]["Wq"], grads["layers"][0]["Wq"]),
        (model.params["layers"][1]["W2"], grads["layers"][1]["W2"]),
        (model.params["wte"], grads["wte"]),
        (model.params["head_W"], grads["head_W"]),
        (model.params["layers"][0]["ln1_g"], grads["layers"][0]["ln1_g"]),
    ]
    for param_array, grad_array in checks:
        _assert_grad_close(model, idx, targets, param_array, grad_array, rng)


def test_training_reduces_loss():
    model = make_model(seed=2)
    rng = np.random.default_rng(7)
    seq_len = model.block_size - 1
    idx = rng.integers(0, model.vocab_size, size=(4, seq_len))
    targets = rng.integers(0, model.vocab_size, size=(4, seq_len))

    first_loss, grads = model.loss_and_grad(idx, targets)
    sgd_step(model.params, grads, lr=0.5)
    for _ in range(50):
        loss, grads = model.loss_and_grad(idx, targets)
        sgd_step(model.params, grads, lr=0.5)
    final_loss, _ = model.loss_and_grad(idx, targets)

    assert final_loss < first_loss


def test_softmax_sums_to_one():
    x = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]])
    probs = softmax(x)
    assert np.allclose(probs.sum(axis=-1), 1.0)


def test_cross_entropy_matches_manual_computation():
    logits = np.array([[[2.0, 0.0, 0.0]]])
    targets = np.array([[0]])
    loss, _ = cross_entropy_loss(logits, targets)
    probs = softmax(logits)
    expected = -np.log(probs[0, 0, 0])
    assert abs(loss - expected) < 1e-9
