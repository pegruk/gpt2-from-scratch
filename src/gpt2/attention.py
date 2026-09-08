from math import sqrt

import torch
import torch.nn as nn
from fancy_einsum import einsum

from gpt2.config import Config


class Attention(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.cfg = cfg
        self.W_Q = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_Q, std=cfg.init_range)
        self.b_Q = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))

        self.W_K = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_K, std=cfg.init_range)
        self.b_K = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))

        self.W_V = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_model, cfg.d_head))
        nn.init.normal_(self.W_V, std=cfg.init_range)
        self.b_V = nn.Parameter(torch.zeros(cfg.n_heads, cfg.d_head))

        self.W_O = nn.Parameter(torch.empty(cfg.n_heads, cfg.d_head, cfg.d_model))
        nn.init.normal_(self.W_O, std=cfg.init_range)
        self.b_O = nn.Parameter(torch.zeros(cfg.d_model))

        self.register_buffer("IGNORE", torch.tensor(float("-inf"), dtype=torch.float32))

    def forward(self, residual_normalized: torch.Tensor) -> torch.Tensor:
        # residual_normalized: [batch, position, d_model]
        q = (
            einsum(
                "batch pos_q d_model, n_heads d_model d_head"
                " -> batch pos_q n_heads d_head",
                residual_normalized,
                self.W_Q,
            )
            + self.b_Q
        )
        k = (
            einsum(
                "batch pos_k d_model, n_heads d_model d_head"
                " -> batch pos_k n_heads d_head",
                residual_normalized,
                self.W_K,
            )
            + self.b_K
        )

        attn_score = einsum(
            "batch pos_q n_heads d_head, batch pos_k n_heads d_head"
            " -> batch n_heads pos_q pos_k",
            q,
            k,
        )
        attn_score = attn_score / sqrt(self.cfg.d_head)
        attn_score = self.causal_mask(attn_score)

        pattern = attn_score.softmax(dim=-1)

        v = (
            einsum(
                "batch pos_k d_model, n_heads d_model d_head"
                " -> batch pos_k n_heads d_head",
                residual_normalized,
                self.W_V,
            )
            + self.b_V
        )

        z = einsum(
            "batch n_heads pos_q pos_k, batch pos_k n_heads d_head"
            " -> batch pos_q n_heads d_head",
            pattern,
            v,
        )

        attn_out = (
            einsum(
                "batch pos_q n_heads d_head, n_heads d_head d_model"
                " -> batch pos_q d_model",
                z,
                self.W_O,
            )
            + self.b_O
        )
        return attn_out

    def causal_mask(self, attn_score: torch.Tensor) -> torch.Tensor:
        mask = torch.triu(
            torch.ones(
                attn_score.size(-2),
                attn_score.size(-1),
                dtype=torch.bool,
                device=attn_score.device,
            ),
            diagonal=1,
        )
        return attn_score.masked_fill(mask, self.get_buffer("IGNORE"))
