import torch
import torch.nn as nn

from gpt2.config import Config
from gpt2.mlp import MLP


def test_mlp_matches_pytorch() -> None:
    torch.manual_seed(42)

    cfg = Config(d_model=16, d_mlp=32)
    custom_mlp = MLP(cfg)
    official_in = nn.Linear(cfg.d_model, cfg.d_mlp)
    official_gelu = nn.GELU(approximate="tanh")
    official_out = nn.Linear(cfg.d_mlp, cfg.d_model)
    residual = torch.randn(2, 5, cfg.d_model)

    with torch.no_grad():
        custom_mlp.W_in.copy_(official_in.weight.T)
        custom_mlp.b_in.copy_(official_in.bias)
        custom_mlp.W_out.copy_(official_out.weight.T)
        custom_mlp.b_out.copy_(official_out.bias)

    actual = custom_mlp(residual)
    expected = official_out(official_gelu(official_in(residual)))

    torch.testing.assert_close(actual, expected)
