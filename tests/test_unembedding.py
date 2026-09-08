import torch
import torch.nn as nn

from gpt2.config import Config
from gpt2.unembedding import Unembed


def test_unembedding_matches_pytorch() -> None:
    torch.manual_seed(42)
    cfg = Config(d_model=16, d_vocab=32)
    custom_unembed = Unembed(cfg)
    official_unembed = nn.Linear(cfg.d_model, cfg.d_vocab, bias=False)

    with torch.no_grad():
        custom_unembed.W_U.copy_(official_unembed.weight)

    residual = torch.randn(2, 5, cfg.d_model)
    actual = custom_unembed(residual)
    expected = official_unembed(residual)
    torch.testing.assert_close(actual, expected)
