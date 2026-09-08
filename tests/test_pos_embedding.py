import torch
import torch.nn as nn

from gpt2.config import Config
from gpt2.pos_embedding import PosEmbed


def test_pos_embedding_matches_pytorch() -> None:
    torch.manual_seed(42)

    cfg = Config(d_model=16, n_ctx=16)

    custom_pos_embed = PosEmbed(cfg)
    official_pos_embed = nn.Embedding(cfg.n_ctx, cfg.d_model)

    with torch.no_grad():
        official_pos_embed.weight.normal_(std=cfg.init_range)
        custom_pos_embed.W_pos.copy_(official_pos_embed.weight)

    seq_len = 10
    positions = torch.arange(seq_len)

    actual = custom_pos_embed(seq_len)
    expected = official_pos_embed(positions)

    torch.testing.assert_close(actual, expected)


def test_pos_embedding_shape() -> None:
    cfg = Config(d_model=16, n_ctx=16)

    pos_embed = PosEmbed(cfg)

    seq_len = 10
    output = pos_embed(seq_len)

    assert output.shape == (seq_len, cfg.d_model)


def test_pos_embedding_uses_correct_positions() -> None:
    cfg = Config(d_model=16, n_ctx=16)

    pos_embed = PosEmbed(cfg)

    seq_len = 3
    output = pos_embed(seq_len)

    torch.testing.assert_close(output[0], pos_embed.W_pos[0])
    torch.testing.assert_close(output[1], pos_embed.W_pos[1])
    torch.testing.assert_close(output[2], pos_embed.W_pos[2])
