import torch
import torch.nn as nn

from gpt2.config import Config
from gpt2.embedding import Embed
from gpt2.layer_norm import LayerNorm
from gpt2.pos_embedding import PosEmbed
from gpt2.transformer_block import TransformerBlock
from gpt2.unembedding import Unembed


class GPT2(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg)
        self.pos_embed = PosEmbed(cfg)
        self.blocks = nn.ModuleList(
            [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.ln_final = LayerNorm(cfg)
        self.unembed = Unembed(cfg)
        self.unembed.W_U = self.embed.W_E

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        # tokens: [batch, position]; output: [batch, position, d_vocab]
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape [batch, position]")
        if not 1 <= tokens.size(1) <= self.cfg.n_ctx:
            raise ValueError("sequence length must be between 1 and n_ctx")

        residual = self.embed(tokens) + self.pos_embed(tokens.size(1))
        for block in self.blocks:
            residual = block(residual)
        return self.unembed(self.ln_final(residual))
