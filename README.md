# GPT-2 from scratch

An educational implementation of GPT-2 components in PyTorch, with explicit
tensor operations and tests against PyTorch's built-in modules.

## Project status

Implemented and tested:

- Token embeddings
- Learned positional embeddings
- Layer normalization
- Causal multi-head self-attention

The MLP, transformer block, unembedding, and complete model files are currently
placeholders. Training, tokenization, pretrained weight loading, and text
generation are not implemented yet.

## Installation

Use Python 3.11 or newer. CI is configured for Python 3.11, 3.12, and 3.13 on Linux.
From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -c requirements/constraints.txt -e '.[dev]'
```

On Windows, activate the environment with `.venv\Scripts\Activate.ps1` in
PowerShell. For a CPU-only Linux environment, install PyTorch before the project:

```bash
python -m pip install -c requirements/constraints.txt torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/constraints.txt -e '.[dev]'
```

For other hardware configurations, follow the
[PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Runtime dependencies are declared in `pyproject.toml`; development tools are
included in the `dev` extra.

`requirements/constraints.txt` records the tested versions of direct dependencies
and development tools. It is not a complete lockfile: pip still resolves their
transitive dependencies. NumPy uses a compatible version range on Python 3.11,
because the version tested locally requires Python 3.12 or newer.
Update these constraints deliberately and rerun the
checks when upgrading dependencies.

## Run the example

```bash
python examples/attention.py
```

This runs embeddings, layer normalization, and attention on synthetic token IDs
with randomly initialized weights. It requires no dataset or model download.

Expected output:

```text
Token shape: (2, 4)
Attention output shape: (2, 4, 16)
```

## Development

With the virtual environment activated, run all checks with `make check`, or run
the equivalent commands individually:

```bash
python -m ruff check src tests examples typings
python -m ruff format --check src tests examples typings
python -m mypy
python -m pyright
python -m pytest
```

Use `make format` to format files and `make build` to build a wheel and source
distribution. Make is optional; `python -m build` is the equivalent build command.

Tests use small configurations and deterministic seeds. Attention tests compare
outputs and gradients against `nn.MultiheadAttention`, check causal isolation,
and exercise masking in float32, float16, and bfloat16. The reduced-precision
tests cover the mask, not the entire forward pass on every device.

Mypy checks the active Python version so installed dependency stubs are analyzed
with the matching interpreter. Both type checkers use a local stub for the
PyTorch `fancy-einsum` API in `typings/`.

## Layout

```text
src/gpt2/       Model components and configuration
tests/          PyTorch comparisons and behavior tests
examples/       Runnable component examples
docs/           Tensor conventions
typings/        Local third-party type declarations
```

See [tensor shapes](docs/tensor-shapes.md) for the axis conventions used throughout
the implemented components. Local checkpoints and generated outputs are ignored
by Git.

## License

[MIT](LICENSE).
