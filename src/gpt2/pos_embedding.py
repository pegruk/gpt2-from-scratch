import torch
import torch.nn as nn

from gpt2.config import Config


class PosEmbed(nn.Module):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        self.W_pos: nn.Parameter = nn.Parameter(torch.empty(cfg.n_ctx, cfg.d_model))
        nn.init.normal_(self.W_pos, std=cfg.init_range)

    def forward(self, len_seq: int) -> torch.Tensor:
        position = torch.arange(len_seq, device=self.W_pos.device)
        pos_embed = self.W_pos[position]

        return pos_embed
