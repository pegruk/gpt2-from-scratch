import torch
import torch.nn as nn

from gpt2.config import Config
from gpt2.transformer_block import TransformerBlock


def test_transformer_block_matches_pytorch() -> None:
    torch.manual_seed(42)
    cfg = Config(d_model=16, d_head=4, n_heads=4, d_mlp=32)
    custom_block = TransformerBlock(cfg)
    official_block = nn.TransformerEncoderLayer(
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        dim_feedforward=cfg.d_mlp,
        dropout=0.0,
        activation=nn.GELU(approximate="tanh"),
        layer_norm_eps=cfg.layer_norm_eps,
        batch_first=True,
        norm_first=True,
    )
    residual = torch.randn(2, 5, cfg.d_model)

    with torch.no_grad():
        attn = official_block.self_attn
        assert attn.in_proj_weight is not None
        assert attn.in_proj_bias is not None
        weights = attn.in_proj_weight.chunk(3)
        biases = attn.in_proj_bias.chunk(3)
        for index, name in enumerate(("Q", "K", "V")):
            getattr(custom_block.attn, f"W_{name}").copy_(
                weights[index]
                .reshape(cfg.n_heads, cfg.d_head, cfg.d_model)
                .transpose(1, 2)
            )
            getattr(custom_block.attn, f"b_{name}").copy_(
                biases[index].reshape(cfg.n_heads, cfg.d_head)
            )
        custom_block.attn.W_O.copy_(
            attn.out_proj.weight.T.reshape(cfg.n_heads, cfg.d_head, cfg.d_model)
        )
        custom_block.attn.b_O.copy_(attn.out_proj.bias)
        custom_block.mlp.W_in.copy_(official_block.linear1.weight.T)
        custom_block.mlp.b_in.copy_(official_block.linear1.bias)
        custom_block.mlp.W_out.copy_(official_block.linear2.weight.T)
        custom_block.mlp.b_out.copy_(official_block.linear2.bias)
        for custom_ln, official_ln in (
            (custom_block.ln1, official_block.norm1),
            (custom_block.ln2, official_block.norm2),
        ):
            official_ln.weight.normal_()
            official_ln.bias.normal_()
            custom_ln.w.copy_(official_ln.weight)
            custom_ln.b.copy_(official_ln.bias)

    mask = custom_block.attn.causal_mask(torch.zeros(5, 5))
    actual = custom_block(residual)
    expected = official_block(residual, src_mask=mask)
    torch.testing.assert_close(actual, expected)
