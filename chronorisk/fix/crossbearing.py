from __future__ import annotations

import torch
from torch import nn

from chronorisk.config import ModelConfig


class CrossAttentionFusion(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_fusion = config.use_fusion
        if config.use_fusion:
            self.ehr_attends_bio = nn.MultiheadAttention(
                config.dim, config.heads, dropout=config.dropout, batch_first=True
            )
            self.bio_attends_ehr = nn.MultiheadAttention(
                config.dim, config.heads, dropout=config.dropout, batch_first=True
            )
            self.fuse = nn.Linear(config.dim * 2, config.dim)
        else:
            self.fuse = nn.Linear(config.dim, config.dim)

    def forward(self, ehr_hidden: torch.Tensor, bio_hidden: torch.Tensor) -> torch.Tensor:
        if not self.use_fusion:
            return self.fuse(ehr_hidden)
        ehr_context, _ = self.ehr_attends_bio(ehr_hidden, bio_hidden, bio_hidden)
        bio_context, _ = self.bio_attends_ehr(bio_hidden, ehr_hidden, ehr_hidden)
        ehr_fused = ehr_hidden + ehr_context
        bio_fused = bio_hidden + bio_context
        return self.fuse(torch.cat([ehr_fused, bio_fused], dim=-1))
