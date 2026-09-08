import torch
import torch.nn.functional as F

from gpt2.config import Config
from gpt2.model import GPT2


def test_model_forward_and_backward() -> None:
    torch.manual_seed(42)
    cfg = Config(d_model=16, d_vocab=32, d_head=4, n_heads=4, d_mlp=32, n_layers=2)
    model = GPT2(cfg)
    tokens = torch.randint(cfg.d_vocab, (2, 6))

    logits = model(tokens)
    assert logits.shape == (2, 6, cfg.d_vocab)
    assert model.unembed.W_U is model.embed.W_E
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.d_vocab), tokens[:, 1:].reshape(-1)
    )
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_model_does_not_use_future_tokens() -> None:
    torch.manual_seed(42)
    cfg = Config(d_model=16, d_vocab=32, d_head=4, n_heads=4, d_mlp=32, n_layers=2)
    model = GPT2(cfg)
    tokens = torch.randint(cfg.d_vocab, (2, 6))
    modified = tokens.clone()
    modified[:, 3:] = (modified[:, 3:] + 1) % cfg.d_vocab

    with torch.no_grad():
        torch.testing.assert_close(model(tokens)[:, :3], model(modified)[:, :3])
