import numpy as np
import pytest

from adapter_constellation.data import build_vocab, decode, encode, get_batches


def test_vocab_roundtrip():
    text = "hello world"
    stoi, itos = build_vocab(text)
    ids = encode(text, stoi)
    assert decode(ids, itos) == text


def test_vocab_is_deterministic_and_sorted():
    text = "banana"
    stoi, _ = build_vocab(text)
    assert list(stoi.keys()) == sorted(set(text))


def test_encode_rejects_unknown_char():
    stoi, _ = build_vocab("abc")
    with pytest.raises(ValueError):
        encode("abcd", stoi)


def test_build_vocab_rejects_empty_text():
    with pytest.raises(ValueError):
        build_vocab("")


def test_get_batches_shapes_and_shift():
    data = np.arange(20, dtype=np.int64)
    rng = np.random.default_rng(0)
    inputs, targets = get_batches(data, block_size=5, batch_size=3, rng=rng)
    assert inputs.shape == (3, 5)
    assert targets.shape == (3, 5)
    # targets are inputs shifted one step to the right
    assert np.array_equal(targets[:, :-1], inputs[:, 1:])
