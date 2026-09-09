# GPT-2 from scratch

A small, educational implementation of the GPT-2 architecture in PyTorch.
The goal is to understand how the components work and fit together, with
explicit tensor operations and readable tests against PyTorch modules.

## What's implemented

- Token and learned positional embeddings
- Layer normalization
- Causal multi-head self-attention
- MLP with the tanh approximation of GELU
- Pre-norm Transformer blocks with residual connections
- A complete model that maps token IDs to vocabulary logits
- Shared token embedding and output projection weights

The default configuration follows GPT-2 small: 12 layers, 12 heads, a hidden
size of 768, and a context length of 1,024 tokens. Use a smaller configuration
when experimenting on CPU.

This is an architecture implementation, not a pretrained text generator. Weights
start randomly initialized. Tokenization, pretrained checkpoint loading, a
training pipeline, and optimized inference are outside the scope of this project.
Dropout and the special GPT-2 residual-projection initialization are omitted
to keep the implementation focused on the forward computation.

## Installation

Requires Python 3.11 or newer. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

On Windows PowerShell, activate with `.venv\Scripts\Activate.ps1` instead.
For a CPU-only PyTorch installation, follow the
[PyTorch instructions](https://pytorch.org/get-started/locally/) before installing
the project.

## Usage

```python
import torch

from gpt2.config import Config
from gpt2.model import GPT2

torch.manual_seed(42)
cfg = Config(
    d_model=32, d_vocab=100, d_head=8, n_heads=4,
    d_mlp=128, n_layers=2, n_ctx=64,
)
model = GPT2(cfg)
tokens = torch.tensor([[1, 5, 9, 2]])

logits = model(tokens)
print(logits.shape)  # torch.Size([1, 4, 100])

# Each position predicts the next token. The last position has no target here.
loss = torch.nn.functional.cross_entropy(
    logits[:, :-1].reshape(-1, cfg.d_vocab),
    tokens[:, 1:].reshape(-1),
)
loss.backward()
```

Input token IDs have shape `[batch, position]`. The model returns raw logits of
shape `[batch, position, d_vocab]`; cross-entropy applies the necessary
normalization. Attention only reads the current token and earlier tokens.

## Tests

```bash
python -m pytest
```

Tests use small configurations and compare the individual components and a
Transformer block with PyTorch equivalents. Model tests check the output shape,
shared weights, gradient propagation, and causal behavior.

Optional formatting checks:

```bash
python -m ruff check src tests
python -m ruff format --check src tests
```

## Structure

```text
src/gpt2/       Components, configuration, and complete model
tests/          Small PyTorch comparisons and model checks
```

Read the components in this order: `embedding.py`, `pos_embedding.py`,
`layer_norm.py`, `attention.py`, `mlp.py`, `transformer_block.py`,
`unembedding.py`, and `model.py`.

## Acknowledgments

This educational project was developed based on Neel Nanda's
two-part Transformer walkthrough:

- [What is a Transformer? (Part 1/2)](https://www.youtube.com/watch?v=bOYE6E8JrtU)
- [Implementing GPT-2 From Scratch (Part 2/2)](https://www.youtube.com/watch?v=dsjUDacBw8o)

The walkthrough served as the foundation for studying and implementing
the GPT-2 architecture in this repository.

## License

[MIT](LICENSE).
