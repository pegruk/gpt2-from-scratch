import torch
import torch.nn as nn
from fancy_einsum import einsum

from gpt2.config import Config


class MLP(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.W_in = nn.Parameter(torch.empty(cfg.d_model, cfg.d_mlp))
        nn.init.normal_(self.W_in, std=cfg.init_range)
        self.b_in = nn.Parameter(torch.zeros(cfg.d_mlp))

        self.W_out = nn.Parameter(torch.empty(cfg.d_mlp, cfg.d_model))
        nn.init.normal_(self.W_out, std=cfg.init_range)
        self.b_out = nn.Parameter(torch.zeros(cfg.d_model))
        self.gelu = nn.GELU(approximate="tanh")

    def forward(self, residual_normalized: torch.Tensor) -> torch.Tensor:
        # residual_normalized: [batch, position, d_model]
        pre = (
            einsum(
                "batch position d_model, d_model d_mlp -> batch position d_mlp",
                residual_normalized,
                self.W_in,
            )
            + self.b_in
        )
        post = self.gelu(pre)
        mlp_out = (
            einsum(
                "batch position d_mlp, d_mlp d_model -> batch position d_model",
                post,
                self.W_out,
            )
            + self.b_out
        )
        return mlp_out
