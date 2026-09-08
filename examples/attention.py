"""Run the implemented components on synthetic token IDs using CPU."""

import torch

from gpt2.attention import Attention
from gpt2.config import Config
from gpt2.embedding import Embed
from gpt2.layer_norm import LayerNorm
from gpt2.pos_embedding import PosEmbed


def main() -> None:
    torch.manual_seed(42)
    cfg = Config(d_model=16, d_vocab=32, d_head=4, n_heads=4, n_ctx=16)
    tokens = torch.tensor([[1, 4, 2, 8], [3, 6, 5, 7]])
    embed = Embed(cfg)
    pos_embed = PosEmbed(cfg)
    layer_norm = LayerNorm(cfg)
    attention = Attention(cfg)

    with torch.no_grad():
        residual = embed(tokens) + pos_embed(tokens.size(1))
        output = attention(layer_norm(residual))

    print(f"Token shape: {tuple(tokens.shape)}")
    print(f"Attention output shape: {tuple(output.shape)}")


if __name__ == "__main__":
    main()
