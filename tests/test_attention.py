import pytest
import torch
import torch.nn as nn

from gpt2.attention import Attention
from gpt2.config import Config


@pytest.mark.parametrize("batch_size, seq_len", [(1, 1), (1, 5), (2, 7)])
def test_attention_matches_pytorch(batch_size: int, seq_len: int) -> None:
    torch.manual_seed(42)

    cfg = Config(d_model=16, d_head=4, n_heads=4)

    custom_attention = Attention(cfg)
    official_attention = nn.MultiheadAttention(
        embed_dim=cfg.d_model,
        num_heads=cfg.n_heads,
        batch_first=True,
    )
    residual = torch.randn(batch_size, seq_len, cfg.d_model, requires_grad=True)
    reference_residual = residual.detach().clone().requires_grad_(True)

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
        reference_residual,
        reference_residual,
        reference_residual,
        attn_mask=causal_mask,
        need_weights=False,
    )

    torch.testing.assert_close(actual, expected)

    output_gradient = torch.randn_like(actual)
    actual.backward(output_gradient)
    expected.backward(output_gradient)
    torch.testing.assert_close(residual.grad, reference_residual.grad)

    assert official_attention.in_proj_weight.grad is not None
    assert official_attention.in_proj_bias.grad is not None
    for custom_weight, reference_gradient in zip(
        (custom_attention.W_Q, custom_attention.W_K, custom_attention.W_V),
        official_attention.in_proj_weight.grad.chunk(3),
        strict=True,
    ):
        torch.testing.assert_close(
            custom_weight.grad,
            reference_gradient.reshape(cfg.n_heads, cfg.d_head, cfg.d_model).transpose(
                1, 2
            ),
        )
    for custom_bias, reference_gradient in zip(
        (custom_attention.b_Q, custom_attention.b_K, custom_attention.b_V),
        official_attention.in_proj_bias.grad.chunk(3),
        strict=True,
    ):
        torch.testing.assert_close(
            custom_bias.grad, reference_gradient.reshape(cfg.n_heads, cfg.d_head)
        )
    assert official_attention.out_proj.weight.grad is not None
    torch.testing.assert_close(
        custom_attention.W_O.grad,
        official_attention.out_proj.weight.grad.T.reshape(
            cfg.n_heads, cfg.d_head, cfg.d_model
        ),
    )
    torch.testing.assert_close(
        custom_attention.b_O.grad, official_attention.out_proj.bias.grad
    )


def test_attention_does_not_use_future_tokens() -> None:
    torch.manual_seed(42)
    attention = Attention(Config(d_model=16, d_head=4, n_heads=4))
    residual = torch.randn(2, 7, 16)
    modified = residual.clone()
    modified[:, 3:] = torch.randn_like(modified[:, 3:]) * 10

    with torch.no_grad():
        original_output = attention(residual)
        modified_output = attention(modified)

    torch.testing.assert_close(original_output[:, :3], modified_output[:, :3])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_attention_causal_mask(dtype: torch.dtype) -> None:
    attention = Attention(Config(d_model=16, d_head=4, n_heads=4)).to(dtype=dtype)
    scores = torch.ones(2, 4, 3, 3, dtype=dtype)

    actual = attention.causal_mask(scores)
    expected = torch.tensor(
        [[1, -torch.inf, -torch.inf], [1, 1, -torch.inf], [1, 1, 1]], dtype=dtype
    )

    torch.testing.assert_close(actual, expected.expand_as(scores))
    torch.testing.assert_close(scores, torch.ones_like(scores))
    assert actual.dtype == dtype
