import torch
import torch.nn as nn

from gpt2.config import Config
from gpt2.layer_norm import LayerNorm


def test_layer_norm_matches_pytorch() -> None:
    torch.manual_seed(42)

    cfg = Config()

    custom_ln = LayerNorm(cfg)
    official_ln = nn.LayerNorm(cfg.d_model, eps=cfg.layer_norm_eps)

    residual = torch.randn(1, 35, cfg.d_model)

    with torch.no_grad():
        official_ln.weight.normal_()
        official_ln.bias.normal_()

        custom_ln.w.copy_(official_ln.weight)
        custom_ln.b.copy_(official_ln.bias)

    actual = custom_ln(residual)
    expected = official_ln(residual)

    torch.testing.assert_close(actual, expected)
