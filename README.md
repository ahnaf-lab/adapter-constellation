# adapter-constellation

LoRA fine-tuning implemented from scratch in numpy, on a tiny character-level
transformer trained on CPU in seconds. The end goal of this project is to
render a LoRA adapter's singular-value spectrum and per-head weight deltas as
a deterministic constellation diagram in the terminal, live, while training
runs -- so that two fine-tunes on different text produce visibly different
constellations, and what an adapter learned can be eyeballed instead of read
off a loss number.

This is being built in milestones. The current milestone implements the base
model everything else sits on top of: a 2-layer character-level transformer
with a full forward and backward pass written by hand in numpy.

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
  LoRA adapters and the live constellation visualization build on top of
  this model in later milestones.

The model is kept behind the `LanguageModel` abstract interface in
`adapter_constellation/model.py`, so later milestones can add other
implementations (or a deterministic fake for tests) without changing calling
code. There is no live API anywhere in this project -- everything here runs
locally on CPU.
