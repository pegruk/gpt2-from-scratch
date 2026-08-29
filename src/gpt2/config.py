from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 768
    d_vocab: int = 50257
    layer_norm_eps: float = 1e-5
    init_range: float = 0.02
