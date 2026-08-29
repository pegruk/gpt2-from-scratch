import torch
import torch.nn as nn
from einops import reduce

from gpt2.config import Config


class LayerNorm(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()

        self.cfg = cfg
        self.w: nn.Parameter = nn.Parameter(torch.ones(cfg.d_model))
        self.b: nn.Parameter = nn.Parameter(torch.zeros(cfg.d_model))

    def forward(self, residual) -> torch.Tensor:
        # residual: [batch, position, d_model]
        mean = reduce(residual, "batch position d_model -> batch position 1", "mean")
        residual_centered = residual - mean

        var = reduce(
            residual_centered.pow(2),
            "batch position d_model -> batch position 1",
            "mean",
        )
        scale = (var + self.cfg.layer_norm_eps).sqrt()

        out = residual_centered / scale

        return out * self.w + self.b
