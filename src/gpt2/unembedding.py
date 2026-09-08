import torch
import torch.nn as nn
from fancy_einsum import einsum

from gpt2.config import Config


class Unembed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        # Same layout as W_E so the full model can share this parameter.
        self.W_U = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_U, std=cfg.init_range)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        return einsum(
            "batch position d_model, d_vocab d_model -> batch position d_vocab",
            residual,
            self.W_U,
        )
