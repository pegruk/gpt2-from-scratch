from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 768
    d_vocab: int = 50257
    d_head: int = 64
    d_mlp: int = 3072
    n_ctx: int = 1024
    n_heads: int = 12
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
