import numpy as np

from adapter_constellation.delta import (
    SpectrumTracker,
    adapter_delta,
    adapter_spectrum,
    delta_svd,
    extract_deltas,
    extract_spectra,
    is_due,
)
from adapter_constellation.lora import init_adapter, lora_sgd_step
from adapter_constellation.lora_model import DEFAULT_TARGETS, LoRATransformer
from adapter_constellation.model import TinyTransformer


def make_base(seed=0):
    return TinyTransformer(
        vocab_size=6, block_size=5, d_model=8, n_heads=2, d_ff=16, n_layers=2, seed=seed
    )


def make_trained_lora(seed=0, rank=3):
    base = make_base(seed=seed)
    lora_model = LoRATransformer(base, rank=rank, alpha=2.0, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    idx = rng.integers(0, base.vocab_size, size=(3, base.block_size - 1))
    targets = rng.integers(0, base.vocab_size, size=(3, base.block_size - 1))
    for _ in range(10):
        loss, grads = lora_model.loss_and_grad(idx, targets)
        lora_sgd_step(lora_model.parameters(), grads, lr=0.5)
    return lora_model


def test_adapter_delta_matches_manual_matmul():
    rng = np.random.default_rng(0)
    adapter = init_adapter(rng, in_dim=5, out_dim=4, rank=2, alpha=2.0)
    adapter["B"] = rng.normal(size=adapter["B"].shape)

    expected = adapter["scale"] * (adapter["A"] @ adapter["B"])
    assert np.allclose(adapter_delta(adapter), expected)


def test_delta_svd_matches_numpy_reference():
    rng = np.random.default_rng(1)
    delta = rng.normal(size=(6, 4))

    result = delta_svd(delta)
    ref_U, ref_S, ref_Vt = np.linalg.svd(delta, full_matrices=False)

    assert np.allclose(result["S"], ref_S)
    assert np.allclose(result["U"], ref_U)
    assert np.allclose(result["Vt"], ref_Vt)
    reconstructed = result["U"] @ np.diag(result["S"]) @ result["Vt"]
    assert np.allclose(reconstructed, delta)


def test_delta_svd_truncates_to_k():
    rng = np.random.default_rng(2)
    delta = rng.normal(size=(6, 4))

    result = delta_svd(delta, k=2)
    assert result["S"].shape == (2,)
    assert result["U"].shape == (6, 2)
    assert result["Vt"].shape == (2, 4)


def test_adapter_spectrum_is_rank_bounded_by_adapter_rank():
    rng = np.random.default_rng(3)
    rank = 2
    adapter = init_adapter(rng, in_dim=8, out_dim=6, rank=rank, alpha=1.0)
    adapter["B"] = rng.normal(size=adapter["B"].shape)

    spectrum = adapter_spectrum(adapter)
    assert spectrum.shape == (6,)
    assert np.all(spectrum[:rank] > 1e-8)
    assert np.all(spectrum[rank:] < 1e-8)


def test_extract_deltas_and_spectra_match_model_shape():
    lora_model = make_trained_lora()

    deltas = extract_deltas(lora_model)
    spectra = extract_spectra(lora_model)

    assert len(deltas) == len(lora_model.adapters) == 2
    assert len(spectra) == 2
    for layer_deltas, layer_spectra, layer_adapters in zip(deltas, spectra, lora_model.adapters):
        assert set(layer_deltas) == set(DEFAULT_TARGETS)
        assert set(layer_spectra) == set(DEFAULT_TARGETS)
        for name in DEFAULT_TARGETS:
            adapter = layer_adapters[name]
            in_dim, out_dim = adapter["A"].shape[0], adapter["B"].shape[1]
            assert layer_deltas[name].shape == (in_dim, out_dim)
            assert np.allclose(layer_deltas[name], adapter_delta(adapter))


def test_is_due_fires_only_on_multiples_and_never_on_zero():
    assert is_due(5, 5) is True
    assert is_due(10, 5) is True
    assert is_due(0, 5) is False
    assert is_due(3, 5) is False


def test_spectrum_tracker_records_every_n_steps():
    lora_model = make_trained_lora()
    tracker = SpectrumTracker(lora_model, every_n=5)

    recorded_steps = []
    for step in range(1, 11):
        snapshot = tracker.maybe_record(step)
        if snapshot is not None:
            recorded_steps.append(snapshot["step"])

    assert recorded_steps == [5, 10]
    assert len(tracker.history) == 2
    assert set(tracker.history[0]["spectra"][0]) == set(DEFAULT_TARGETS)
