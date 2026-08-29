import torch
import torch.nn as nn

from gpt2.config import Config


class Embed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.W_E: nn.Parameter = nn.Parameter(torch.empty(cfg.d_vocab, cfg.d_model))
        nn.init.normal_(self.W_E, std=cfg.init_range)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.W_E[tokens]
