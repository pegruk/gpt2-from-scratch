import torch
import torch.nn as nn

from gpt2.attention import Attention
from gpt2.config import Config


def test_attention_matches_pytorch() -> None:
    torch.manual_seed(42)

    cfg = Config(d_model=16, d_head=4, n_heads=4)

    custom_attention = Attention(cfg)
    official_attention = nn.MultiheadAttention(
        embed_dim=cfg.d_model,
        num_heads=cfg.n_heads,
        batch_first=True,
    )
    residual = torch.randn(2, 5, cfg.d_model)

    with torch.no_grad():
        assert official_attention.in_proj_weight is not None
        assert official_attention.in_proj_bias is not None
        assert official_attention.out_proj.bias is not None
        official_attention.in_proj_bias.normal_(std=cfg.init_range)
        official_attention.out_proj.bias.normal_(std=cfg.init_range)
        q_weight, k_weight, v_weight = official_attention.in_proj_weight.chunk(3)
        q_bias, k_bias, v_bias = official_attention.in_proj_bias.chunk(3)

        # PyTorch stores each projection as [n_heads * d_head, d_model].
        custom_attention.W_Q.copy_(
            q_weight.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        )
        custom_attention.W_K.copy_(
            k_weight.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        )
        custom_attention.W_V.copy_(
            v_weight.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(1, 2)
        )
        custom_attention.b_Q.copy_(q_bias.reshape(cfg.n_heads, cfg.d_head))
        custom_attention.b_K.copy_(k_bias.reshape(cfg.n_heads, cfg.d_head))
        custom_attention.b_V.copy_(v_bias.reshape(cfg.n_heads, cfg.d_head))
        custom_attention.W_O.copy_(
            official_attention.out_proj.weight.T.reshape(
                cfg.n_heads, cfg.d_head, cfg.d_model
            )
        )
        custom_attention.b_O.copy_(official_attention.out_proj.bias)

    causal_mask = custom_attention.causal_mask(
        torch.zeros(residual.size(1), residual.size(1))
    )
    actual = custom_attention(residual)
    expected, _ = official_attention(
        residual,
        residual,
        residual,
        attn_mask=causal_mask,
        need_weights=False,
    )

    torch.testing.assert_close(actual, expected)
