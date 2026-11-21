"""A tiny 2-layer character-level transformer, implemented from scratch in numpy.

No autograd: every forward op has a matching, hand-derived backward op, wired
together in `TinyTransformer.loss_and_grad`. The model sits behind the
`LanguageModel` interface so later milestones (LoRA adapters, training loops)
can swap in other implementations without touching the rest of the codebase;
tests only ever exercise this local, deterministic numpy model.
"""

from __future__ import annotations

import abc

import numpy as np

Array = np.ndarray


# ---------------------------------------------------------------------------
# Low-level ops: each `*_forward` returns (output, cache); each matching
# `*_backward` takes (upstream_grad, cache) and returns the gradients needed
# to continue the chain rule backward through that op.
# ---------------------------------------------------------------------------


def softmax(x: Array, axis: int = -1) -> Array:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def layer_norm_forward(x: Array, gamma: Array, beta: Array, eps: float = 1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    std_inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * std_inv
    out = xhat * gamma + beta
    cache = (xhat, std_inv, gamma)
    return out, cache


def layer_norm_backward(dout: Array, cache):
    xhat, std_inv, gamma = cache
    reduce_axes = tuple(range(dout.ndim - 1))
    dgamma = np.sum(dout * xhat, axis=reduce_axes)
    dbeta = np.sum(dout, axis=reduce_axes)
    dxhat = dout * gamma
    mean_dxhat = dxhat.mean(axis=-1, keepdims=True)
    mean_dxhat_xhat = (dxhat * xhat).mean(axis=-1, keepdims=True)
    dx = std_inv * (dxhat - mean_dxhat - xhat * mean_dxhat_xhat)
    return dx, dgamma, dbeta


def linear_forward(x: Array, W: Array, b: Array):
    out = x @ W + b
    cache = (x, W)
    return out, cache


def linear_backward(dout: Array, cache):
    x, W = cache
    in_dim = x.shape[-1]
    out_dim = dout.shape[-1]
    x_flat = x.reshape(-1, in_dim)
    dout_flat = dout.reshape(-1, out_dim)
    dW = x_flat.T @ dout_flat
    db = dout_flat.sum(axis=0)
    dx = dout @ W.T
    return dx, dW, db


def relu_forward(x: Array):
    cache = x > 0
    return x * cache, cache


def relu_backward(dout: Array, cache):
    return dout * cache


def cross_entropy_loss(logits: Array, targets: Array):
    """Softmax cross-entropy over the last axis, averaged over all other axes.

    Returns (scalar loss, dlogits) where dlogits is the exact gradient of the
    mean loss with respect to `logits` -- the combined softmax+NLL backward.
    """
    *lead, vocab_size = logits.shape
    n = int(np.prod(lead)) if lead else 1
    logits_flat = logits.reshape(n, vocab_size)
    targets_flat = targets.reshape(n)
    probs_flat = softmax(logits_flat, axis=-1)
    correct = probs_flat[np.arange(n), targets_flat]
    loss = -np.mean(np.log(correct + 1e-12))
    dlogits_flat = probs_flat.copy()
    dlogits_flat[np.arange(n), targets_flat] -= 1.0
    dlogits_flat /= n
    return loss, dlogits_flat.reshape(logits.shape)


def causal_self_attention_forward(x: Array, p: dict, n_heads: int):
    B, T, C = x.shape
    head_size = C // n_heads

    q, cache_q = linear_forward(x, p["Wq"], p["bq"])
    k, cache_k = linear_forward(x, p["Wk"], p["bk"])
    v, cache_v = linear_forward(x, p["Wv"], p["bv"])

    def split_heads(t: Array) -> Array:
        return t.reshape(B, T, n_heads, head_size).transpose(0, 2, 1, 3)

    qh, kh, vh = split_heads(q), split_heads(k), split_heads(v)

    scores = (qh @ kh.transpose(0, 1, 3, 2)) / np.sqrt(head_size)
    causal_mask = np.triu(np.ones((T, T), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)
    attn = softmax(scores, axis=-1)
    out_heads = attn @ vh  # (B, n_heads, T, head_size)
    out = out_heads.transpose(0, 2, 1, 3).reshape(B, T, C)

    y, cache_o = linear_forward(out, p["Wo"], p["bo"])
    cache = {
        "cache_q": cache_q,
        "cache_k": cache_k,
        "cache_v": cache_v,
        "cache_o": cache_o,
        "qh": qh,
        "kh": kh,
        "vh": vh,
        "attn": attn,
        "shape": (B, T, C, n_heads, head_size),
    }
    return y, cache


def causal_self_attention_backward(dy: Array, cache: dict):
    B, T, C, n_heads, head_size = cache["shape"]
    attn, qh, kh, vh = cache["attn"], cache["qh"], cache["kh"], cache["vh"]

    dout, dWo, dbo = linear_backward(dy, cache["cache_o"])
    dout_heads = dout.reshape(B, T, n_heads, head_size).transpose(0, 2, 1, 3)

    dattn = dout_heads @ vh.transpose(0, 1, 3, 2)
    dvh = attn.transpose(0, 1, 3, 2) @ dout_heads

    dmasked = attn * (dattn - np.sum(dattn * attn, axis=-1, keepdims=True))
    draw = dmasked / np.sqrt(head_size)

    dqh = draw @ kh
    dkh = draw.transpose(0, 1, 3, 2) @ qh

    def merge_heads(t: Array) -> Array:
        return t.transpose(0, 2, 1, 3).reshape(B, T, C)

    dq, dk, dv = merge_heads(dqh), merge_heads(dkh), merge_heads(dvh)

    dx_q, dWq, dbq = linear_backward(dq, cache["cache_q"])
    dx_k, dWk, dbk = linear_backward(dk, cache["cache_k"])
    dx_v, dWv, dbv = linear_backward(dv, cache["cache_v"])
    dx = dx_q + dx_k + dx_v

    grads = {
        "Wq": dWq, "bq": dbq,
        "Wk": dWk, "bk": dbk,
        "Wv": dWv, "bv": dbv,
        "Wo": dWo, "bo": dbo,
    }
    return dx, grads


def mlp_forward(x: Array, p: dict):
    h, cache1 = linear_forward(x, p["W1"], p["b1"])
    a, cache_relu = relu_forward(h)
    y, cache2 = linear_forward(a, p["W2"], p["b2"])
    return y, (cache1, cache_relu, cache2)


def mlp_backward(dy: Array, cache):
    cache1, cache_relu, cache2 = cache
    da, dW2, db2 = linear_backward(dy, cache2)
    dh = relu_backward(da, cache_relu)
    dx, dW1, db1 = linear_backward(dh, cache1)
    return dx, {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def block_forward(x: Array, p: dict, n_heads: int):
    ln1_out, ln1_cache = layer_norm_forward(x, p["ln1_g"], p["ln1_b"])
    attn_out, attn_cache = causal_self_attention_forward(ln1_out, p, n_heads)
    x2 = x + attn_out

    ln2_out, ln2_cache = layer_norm_forward(x2, p["ln2_g"], p["ln2_b"])
    mlp_out, mlp_cache = mlp_forward(ln2_out, p)
    x3 = x2 + mlp_out

    cache = {
        "ln1_cache": ln1_cache,
        "attn_cache": attn_cache,
        "ln2_cache": ln2_cache,
        "mlp_cache": mlp_cache,
    }
    return x3, cache


def block_backward(dx3: Array, cache: dict):
    dx2_from_mlp, mlp_grads = mlp_backward(dx3, cache["mlp_cache"])
    dln2_out, dln2_g, dln2_b = layer_norm_backward(dx2_from_mlp, cache["ln2_cache"])
    dx2 = dx3 + dln2_out

    dx_from_attn, attn_grads = causal_self_attention_backward(dx2, cache["attn_cache"])
    dln1_out, dln1_g, dln1_b = layer_norm_backward(dx_from_attn, cache["ln1_cache"])
    dx = dx2 + dln1_out

    grads = {
        "ln1_g": dln1_g, "ln1_b": dln1_b,
        "ln2_g": dln2_g, "ln2_b": dln2_b,
        **attn_grads,
        **mlp_grads,
    }
    return dx, grads


# ---------------------------------------------------------------------------
# Model interface and the tiny transformer that implements it.
# ---------------------------------------------------------------------------


class LanguageModel(abc.ABC):
    """Minimal interface a character-level model must implement.

    Keeping training code behind this interface means the rest of the
    project (and its tests) never depends on a concrete model: swapping in a
    different architecture, or a deterministic fake for a test, only
    requires implementing these three methods.
    """

    @abc.abstractmethod
    def forward(self, idx: Array):
        """Returns logits of shape (batch, seq_len, vocab_size)."""

    @abc.abstractmethod
    def loss_and_grad(self, idx: Array, targets: Array):
        """Returns (scalar loss, gradient tree matching `self.params`)."""

    @abc.abstractmethod
    def parameters(self) -> dict:
        """Returns the model's parameter tree (nested dicts/lists of ndarrays)."""


def _init_linear(rng: np.random.Generator, in_dim: int, out_dim: int, scale: float = 0.02):
    W = rng.normal(0.0, scale, size=(in_dim, out_dim))
    b = np.zeros(out_dim)
    return W, b


class TinyTransformer(LanguageModel):
    """A 2-layer causal transformer over characters, forward+backward in numpy."""

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        d_model: int = 32,
        n_heads: int = 4,
        d_ff: int = 64,
        n_layers: int = 2,
        seed: int = 0,
    ):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers

        rng = np.random.default_rng(seed)
        layers = []
        for _ in range(n_layers):
            Wq, bq = _init_linear(rng, d_model, d_model)
            Wk, bk = _init_linear(rng, d_model, d_model)
            Wv, bv = _init_linear(rng, d_model, d_model)
            Wo, bo = _init_linear(rng, d_model, d_model)
            W1, b1 = _init_linear(rng, d_model, d_ff)
            W2, b2 = _init_linear(rng, d_ff, d_model)
            layers.append({
                "ln1_g": np.ones(d_model), "ln1_b": np.zeros(d_model),
                "Wq": Wq, "bq": bq, "Wk": Wk, "bk": bk,
                "Wv": Wv, "bv": bv, "Wo": Wo, "bo": bo,
                "ln2_g": np.ones(d_model), "ln2_b": np.zeros(d_model),
                "W1": W1, "b1": b1, "W2": W2, "b2": b2,
            })

        head_W, head_b = _init_linear(rng, d_model, vocab_size)
        self.params = {
            "wte": rng.normal(0.0, 0.02, size=(vocab_size, d_model)),
            "wpe": rng.normal(0.0, 0.02, size=(block_size, d_model)),
            "layers": layers,
            "lnf_g": np.ones(d_model),
            "lnf_b": np.zeros(d_model),
            "head_W": head_W,
            "head_b": head_b,
        }

    def parameters(self) -> dict:
        return self.params

    def forward(self, idx: Array):
        logits, _ = self._forward_with_cache(idx)
        return logits

    def _forward_with_cache(self, idx: Array):
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.block_size}")

        tok_emb = self.params["wte"][idx]
        pos_emb = self.params["wpe"][:T]
        x = tok_emb + pos_emb

        layer_caches = []
        for layer_params in self.params["layers"]:
            x, cache = block_forward(x, layer_params, self.n_heads)
            layer_caches.append(cache)

        x, lnf_cache = layer_norm_forward(x, self.params["lnf_g"], self.params["lnf_b"])
        logits, head_cache = linear_forward(x, self.params["head_W"], self.params["head_b"])

        cache = {
            "idx": idx,
            "T": T,
            "layer_caches": layer_caches,
            "lnf_cache": lnf_cache,
            "head_cache": head_cache,
        }
        return logits, cache

    def _backward(self, dlogits: Array, cache: dict) -> dict:
        dx, dhead_W, dhead_b = linear_backward(dlogits, cache["head_cache"])
        dx, dlnf_g, dlnf_b = layer_norm_backward(dx, cache["lnf_cache"])

        layer_grads = []
        for layer_cache in reversed(cache["layer_caches"]):
            dx, grads = block_backward(dx, layer_cache)
            layer_grads.append(grads)
        layer_grads.reverse()

        idx, T = cache["idx"], cache["T"]
        dwte = np.zeros_like(self.params["wte"])
        np.add.at(dwte, idx, dx)
        dwpe = np.zeros_like(self.params["wpe"])
        dwpe[:T] += dx.sum(axis=0)

        return {
            "wte": dwte,
            "wpe": dwpe,
            "layers": layer_grads,
            "lnf_g": dlnf_g,
            "lnf_b": dlnf_b,
            "head_W": dhead_W,
            "head_b": dhead_b,
        }

    def loss_and_grad(self, idx: Array, targets: Array):
        logits, cache = self._forward_with_cache(idx)
        loss, dlogits = cross_entropy_loss(logits, targets)
        grads = self._backward(dlogits, cache)
        return loss, grads


def sgd_step(params, grads, lr: float) -> None:
    """Applies an in-place SGD update to a parameter tree using matching grads."""
    if isinstance(params, dict):
        for key in params:
            sgd_step(params[key], grads[key], lr)
    elif isinstance(params, list):
        for p_item, g_item in zip(params, grads):
            sgd_step(p_item, g_item, lr)
    elif isinstance(params, np.ndarray):
        params -= lr * grads
    else:
        raise TypeError(f"unsupported parameter tree node: {type(params)!r}")
