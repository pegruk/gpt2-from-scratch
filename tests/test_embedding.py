import torch
import torch.nn as nn

from gpt2.config import Config
from gpt2.embedding import Embed


def test_embedding_matches_pytorch() -> None:
    torch.manual_seed(42)

    cfg = Config()

    custom_embed = Embed(cfg)
    official_embed = nn.Embedding(cfg.d_vocab, cfg.d_model)

    with torch.no_grad():
        official_embed.weight.normal_()

        custom_embed.W_E.copy_(official_embed.weight)

    tokens = torch.arange(start=0, end=10)

    actual = custom_embed(tokens)
    expected = official_embed(tokens)

    torch.testing.assert_close(actual, expected)
