"""LoRA adapters: frozen base weights plus a trainable low-rank update.

For a frozen weight matrix `W` of shape `(in_dim, out_dim)`, a LoRA adapter
adds `scale * A @ B` on top of it, where `A` is `(in_dim, rank)` and `B` is
`(rank, out_dim)`. Only `A` and `B` ever receive a gradient or get updated;
`W` is only ever read, never trained.

Because `effective_weight(W, adapter) == W + scale * A @ B` has exactly the
same shape as `W`, injecting a LoRA adapter into the transformer from
milestone 1 does not require touching its forward/backward math at all:
`lora_model.py` substitutes the effective weight into the existing
`linear_forward`/`block_forward` machinery, and afterward converts the
gradient that backward pass produces for the *effective* weight into
gradients for `A` and `B` via `adapter_grad`.
"""

from __future__ import annotations

import numpy as np

Array = np.ndarray


def init_adapter(rng: np.random.Generator, in_dim: int, out_dim: int, rank: int, alpha: float = 1.0) -> dict:
    """Creates a LoRA adapter for a weight of shape `(in_dim, out_dim)`.

    `A` is initialized with small random noise and `B` with zeros, so the
    adapter starts as a no-op (`A @ B == 0`) and the base model's behavior
    is unchanged until training moves `B` away from zero -- the standard
    LoRA initialization. `scale` follows the usual `alpha / rank`
    convention, so adapter magnitude is comparable across different ranks.
    """
    if rank < 1:
        raise ValueError("rank must be >= 1")
    A = rng.normal(0.0, 0.02, size=(in_dim, rank))
    B = np.zeros((rank, out_dim))
    return {"A": A, "B": B, "scale": alpha / rank}


def effective_weight(W: Array, adapter: dict) -> Array:
    """Returns `W + scale * A @ B`, the weight the forward pass actually uses."""
    return W + adapter["scale"] * (adapter["A"] @ adapter["B"])


def adapter_grad(dW_eff: Array, adapter: dict) -> dict:
    """Converts a gradient w.r.t. the effective weight into grads for A, B.

    `effective_weight` is `W + scale * A @ B`, so by the product rule:
        d/dA = scale * dW_eff @ B.T
        d/dB = scale * A.T @ dW_eff
    `W` itself never receives a gradient here -- it is frozen.
    """
    scale, A, B = adapter["scale"], adapter["A"], adapter["B"]
    dA = scale * dW_eff @ B.T
    dB = scale * A.T @ dW_eff
    return {"A": dA, "B": dB}


def lora_sgd_step(adapters, grads, lr: float) -> None:
    """Applies an in-place SGD update to `A` and `B` only; `scale` is fixed.

    `adapters` is a nested structure of dicts/lists bottoming out in adapter
    dicts (each with `A`, `B`, `scale`); `grads` must have the same shape,
    bottoming out in dicts with `A`, `B`.
    """
    if isinstance(adapters, dict) and "A" in adapters and "B" in adapters:
        adapters["A"] -= lr * grads["A"]
        adapters["B"] -= lr * grads["B"]
    elif isinstance(adapters, dict):
        for key in adapters:
            lora_sgd_step(adapters[key], grads[key], lr)
    elif isinstance(adapters, list):
        for a_item, g_item in zip(adapters, grads):
            lora_sgd_step(a_item, g_item, lr)
    else:
        raise TypeError(f"unsupported adapter tree node: {type(adapters)!r}")
