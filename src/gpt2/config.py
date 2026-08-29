from dataclasses import dataclass


@dataclass
class Config:
    d_model: int = 768
    layer_norm_eps: float = 1e-5
