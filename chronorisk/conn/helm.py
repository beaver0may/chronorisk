from __future__ import annotations

import torch
from torch import nn

from chronorisk.config import ModelConfig
from chronorisk.datum import N_COMPLICATIONS, N_TIERS


class HelmPolicy(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(config.dim, config.dim),
            nn.GELU(),
            nn.Linear(config.dim, N_COMPLICATIONS * N_TIERS),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raw = self.head(state)
        return raw.reshape(*state.shape[:-1], N_COMPLICATIONS, N_TIERS)
