"""LoRA-adapted transformer: base weights frozen, adapters trained.

`LoRATransformer` wraps a `TinyTransformer` and injects a LoRA adapter (see
`lora.py`) on a chosen subset of its weight matrices. The base model's own
forward/backward functions (`block_forward`, `block_backward`, ...) are
reused completely unchanged: on every forward pass each targeted weight `W`
is swapped for `W + scale * A @ B` before calling them, and on the backward
pass the gradient they produce for that effective weight is converted into
gradients for `A` and `B` only -- the base weight is read but never
written, so it never moves and stays shared with whatever other adapter is
trained from it next.
"""

from __future__ import annotations

import numpy as np

from .lora import adapter_grad, effective_weight, init_adapter
from .model import (
    LanguageModel,
    TinyTransformer,
    block_backward,
    block_forward,
    cross_entropy_loss,
    layer_norm_backward,
    layer_norm_forward,
    linear_backward,
    linear_forward,
)

Array = np.ndarray

# The standard LoRA target set: the four attention projections and the two
# MLP projections. `head_W` (the output vocabulary projection) and the
# embeddings are left frozen with no adapter, as is typical.
DEFAULT_TARGETS = ("Wq", "Wk", "Wv", "Wo", "W1", "W2")


class LoRATransformer(LanguageModel):
    """A `TinyTransformer` with frozen base weights and trainable LoRA adapters.

    `parameters()` and `loss_and_grad()` only ever expose the adapters'
    `A`/`B` matrices, so an optimizer driven off this interface can never
    touch the base model's weights.
    """

    def __init__(
        self,
        base: TinyTransformer,
        rank: int = 4,
        alpha: float = 1.0,
        targets: tuple[str, ...] = DEFAULT_TARGETS,
        seed: int = 0,
    ):
        self.base = base
        self.rank = rank
        self.targets = targets

        rng = np.random.default_rng(seed)
        self.adapters = []
        for layer_params in base.params["layers"]:
            layer_adapters = {}
            for name in targets:
                in_dim, out_dim = layer_params[name].shape
                layer_adapters[name] = init_adapter(rng, in_dim, out_dim, rank, alpha)
            self.adapters.append(layer_adapters)

    def parameters(self) -> dict:
        """Returns only the trainable LoRA parameters (`A`, `B` per target, per layer)."""
        return {"layers": self.adapters}

    def _effective_layer_params(self, layer_params: dict, layer_adapters: dict) -> dict:
        effective = dict(layer_params)
        for name, adapter in layer_adapters.items():
            effective[name] = effective_weight(layer_params[name], adapter)
        return effective

    def forward(self, idx: Array) -> Array:
        logits, _ = self._forward_with_cache(idx)
        return logits

    def _forward_with_cache(self, idx: Array):
        base = self.base
        _, T = idx.shape
        if T > base.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {base.block_size}")

        tok_emb = base.params["wte"][idx]
        pos_emb = base.params["wpe"][:T]
        x = tok_emb + pos_emb

        layer_caches = []
        for layer_params, layer_adapters in zip(base.params["layers"], self.adapters):
            effective_params = self._effective_layer_params(layer_params, layer_adapters)
            x, cache = block_forward(x, effective_params, base.n_heads)
            layer_caches.append(cache)

        x, lnf_cache = layer_norm_forward(x, base.params["lnf_g"], base.params["lnf_b"])
        logits, head_cache = linear_forward(x, base.params["head_W"], base.params["head_b"])

        cache = {"layer_caches": layer_caches, "lnf_cache": lnf_cache, "head_cache": head_cache}
        return logits, cache

    def _backward(self, dlogits: Array, cache: dict) -> dict:
        dx, _dhead_W, _dhead_b = linear_backward(dlogits, cache["head_cache"])
        dx, _dlnf_g, _dlnf_b = layer_norm_backward(dx, cache["lnf_cache"])

        layer_grads = []
        for layer_cache, layer_adapters in zip(reversed(cache["layer_caches"]), reversed(self.adapters)):
            dx, grads = block_backward(dx, layer_cache)
            layer_grads.append({name: adapter_grad(grads[name], layer_adapters[name]) for name in layer_adapters})
        layer_grads.reverse()

        return {"layers": layer_grads}

    def loss_and_grad(self, idx: Array, targets: Array):
        logits, cache = self._forward_with_cache(idx)
        loss, dlogits = cross_entropy_loss(logits, targets)
        grads = self._backward(dlogits, cache)
        return loss, grads
