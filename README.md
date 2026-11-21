# adapter-constellation

LoRA fine-tuning implemented from scratch in numpy, on a tiny character-level
transformer trained on CPU in seconds. The end goal of this project is to
render a LoRA adapter's singular-value spectrum and per-head weight deltas as
a deterministic constellation diagram in the terminal, live, while training
runs -- so that two fine-tunes on different text produce visibly different
constellations, and what an adapter learned can be eyeballed instead of read
off a loss number.

This is being built in milestones. Milestone 1 implemented the base model
everything else sits on top of: a 2-layer character-level transformer with a
full forward and backward pass written by hand in numpy. Milestone 2 added
LoRA: low-rank `A`/`B` adapters injected per weight matrix, with the base
model frozen and only the adapters trained by gradient descent. Milestone 3
extracted, from each adapter, exactly the data the constellation diagram
will eventually draw: the singular-value spectrum of its effective weight
delta. The current milestone turns that data into the diagram's actual
coordinates: a deterministic map from singular values and per-head deltas to
each star's position and brightness.

## Install

Requires Python 3.10+.

```bash
pip install -r requirements.txt
```

The only dependency is `numpy`, used for array math and nothing else.
Writing a transformer's matrix multiplies, softmax, and layer norm by hand in
pure Python would be slow and would not change anything about what the
project demonstrates, so numpy stands in for the standard library here.

## Usage

The model and dataset preparation are plain Python modules; there is no CLI
yet (that lands in a later milestone alongside the training loop). For now,
exercise them directly:

```python
import numpy as np

from adapter_constellation.data import build_vocab, encode, get_batches
from adapter_constellation.model import TinyTransformer, sgd_step

text = "the quick brown fox jumps over the lazy dog"
stoi, itos = build_vocab(text)
data = encode(text, stoi)

block_size = 8
model = TinyTransformer(vocab_size=len(stoi), block_size=block_size)

rng = np.random.default_rng(0)
inputs, targets = get_batches(data, block_size=block_size, batch_size=4, rng=rng)

loss, grads = model.loss_and_grad(inputs, targets)
sgd_step(model.params, grads, lr=0.1)
print(loss)
```

To fine-tune with LoRA instead of training the full model, wrap a trained
(or freshly initialized) `TinyTransformer` in a `LoRATransformer`. The base
weights are read every forward pass but never written; `parameters()` and
`loss_and_grad()` only ever expose the adapters' `A`/`B` matrices, so
`lora_sgd_step` can only ever update those:

```python
from adapter_constellation.lora import lora_sgd_step
from adapter_constellation.lora_model import LoRATransformer

lora_model = LoRATransformer(model, rank=4, alpha=2.0)

for _ in range(100):
    inputs, targets = get_batches(data, block_size=block_size, batch_size=4, rng=rng)
    loss, grads = lora_model.loss_and_grad(inputs, targets)
    lora_sgd_step(lora_model.parameters(), grads, lr=0.5)
print(loss)
```

Every so often during training, you probably want to look at what an
adapter's low-rank update actually *is*, rather than just the scalar loss.
`adapter_constellation/delta.py` turns each adapter into its effective delta
(`scale * A @ B`) and that delta's singular-value spectrum; `SpectrumTracker`
hooks into a training loop and records a snapshot every `every_n` steps:

```python
from adapter_constellation.delta import SpectrumTracker

tracker = SpectrumTracker(lora_model, every_n=10)
for step in range(1, 101):
    inputs, targets = get_batches(data, block_size=block_size, batch_size=4, rng=rng)
    loss, grads = lora_model.loss_and_grad(inputs, targets)
    lora_sgd_step(lora_model.parameters(), grads, lr=0.5)
    tracker.maybe_record(step)

# One entry per recorded step; each holds every layer's/target's spectrum.
print([snapshot["step"] for snapshot in tracker.history])
```

Finally, `adapter_constellation/constellation.py` turns that same data into
the diagram's actual geometry: a list of "stars," each with an `(x, y)`
position in the unit disk and a `brightness` in `[0, 1]`. Every singular
value gets a star, and every attention projection (`Wq`/`Wk`/`Wv`/`Wo`) also
gets one star per attention head, sized by how much that head's slice of the
delta moved. A star's position depends only on its own identity (which
layer, target, singular value or head) via a hash, never on the trained
numbers -- so the same adapter shape always draws stars in the same spots,
and only brightness changes as training proceeds:

```python
from adapter_constellation.constellation import build_constellation

stars = build_constellation(lora_model, k=4)
brightest = max(stars, key=lambda s: s["brightness"])
print(brightest["layer"], brightest["target"], brightest["kind"], brightest["brightness"])
```

Run the test suite with:

```bash
python -m pytest
```

## Status

Built autonomously, one milestone at a time. Each milestone is only kept if
its automated tests pass; nothing here was hand-verified by eyeballing output.

## What this demonstrates

- **Fine-tuning primitives (dataset prep, training/eval loop groundwork on a
  tiny model).** `adapter_constellation/data.py` turns raw text into a
  character vocabulary and batches of `(input, target)` windows
  (`build_vocab`, `encode`/`decode`, `get_batches`). `adapter_constellation/model.py`
  implements the model those batches are trained on: `TinyTransformer.loss_and_grad`
  runs a forward pass, computes cross-entropy loss, and backpropagates
  gradients through every layer by hand (no autograd library), and
  `sgd_step` applies them. `tests/test_model.py::test_training_reduces_loss`
  is a minimal training loop that checks loss actually goes down.
  The live constellation visualization builds on top of this in a later
  milestone.
- **LoRA / adapters.** `adapter_constellation/lora.py` implements the
  low-rank adapter itself: `init_adapter` creates a pair of matrices `A`
  (random) and `B` (zero), `effective_weight` computes `W + scale * A @ B`,
  and `adapter_grad` derives `dA`/`dB` from the gradient of that effective
  weight via the product rule. `adapter_constellation/lora_model.py`'s
  `LoRATransformer` injects an adapter onto each targeted weight matrix
  (`Wq`, `Wk`, `Wv`, `Wo`, `W1`, `W2`) in every layer of a `TinyTransformer`,
  reusing that model's own forward/backward functions unchanged by
  substituting the effective weight in before calling them. Its
  `parameters()`/`loss_and_grad()` only ever expose the adapters, so the
  base model is read during training but structurally cannot be written to;
  `lora_sgd_step` updates `A`/`B` by gradient descent.
  `tests/test_lora.py::test_base_weights_are_never_modified_by_training` and
  `::test_training_reduces_loss` check both halves of that: the base stays
  frozen, and the adapter still learns.

- **Delta extraction for inspecting what an adapter learned.**
  `adapter_constellation/delta.py`'s `adapter_delta` computes an adapter's
  actual contribution to its weight (`scale * A @ B`), and `delta_svd` runs
  that through `numpy.linalg.svd` to get its singular-value spectrum --
  bounded in rank by the adapter's own rank, since `A @ B` can never exceed
  it. `extract_deltas`/`extract_spectra` do this for every target in every
  layer of a `LoRATransformer`, and `SpectrumTracker` hooks into a training
  loop to record a snapshot only every N steps (`is_due`) instead of every
  step. `tests/test_delta.py::test_delta_svd_matches_numpy_reference` checks
  the decomposition against numpy's own SVD and reconstructs the original
  delta from it; `::test_adapter_spectrum_is_rank_bounded_by_adapter_rank`
  checks that singular values past the adapter's rank are numerically zero.
  This is the exact data the live constellation diagram will render in a
  later milestone.

- **Constellation mapping from adapter deltas to star geometry.**
  `adapter_constellation/constellation.py`'s `star_position` hashes a star's
  identity (`layer`, `target`, `kind`, `index`) into a deterministic point in
  the unit disk, so the same adapter shape always places its stars the same
  way regardless of the trained values. `spectrum_stars` turns each
  adapter's singular-value spectrum into one star per value, and
  `head_delta_magnitudes`/`head_stars` split an attention projection's delta
  into per-head blocks (mirroring the head split in `model.py`'s attention)
  and turn each block's norm into a star; brightness is each magnitude
  normalized against the brightest star in its own group, so it always lands
  in `[0, 1]`. `build_constellation` assembles every layer's and target's
  stars for a `LoRATransformer` in one call.
  `tests/test_constellation.py::test_star_position_depends_on_identity_not_on_magnitude`
  checks that position tracks identity and not the trained numbers;
  `::test_build_constellation_is_deterministic_for_the_same_model_state` and
  `::test_head_delta_magnitudes_matches_manual_block_norms` check
  reproducibility and the per-head math respectively. This list of stars is
  exactly what the live terminal diagram will draw in a later milestone.

The model is kept behind the `LanguageModel` abstract interface in
`adapter_constellation/model.py`, so later milestones can add other
implementations (or a deterministic fake for tests) without changing calling
code. There is no live API anywhere in this project -- everything here runs
locally on CPU.
