import torch
import torch.nn as nn

from gpt2.attention import Attention
from gpt2.config import Config
from gpt2.layer_norm import LayerNorm
from gpt2.mlp import MLP


class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.ln1 = LayerNorm(cfg)
        self.attn = Attention(cfg)
        self.ln2 = LayerNorm(cfg)
        self.mlp = MLP(cfg)

    def forward(self, residual: torch.Tensor) -> torch.Tensor:
        # residual: [batch, position, d_model]
        residual = residual + self.attn(self.ln1(residual))
        return residual + self.mlp(self.ln2(residual))
