from __future__ import annotations

import torch
from torch import nn


class TimeAwarePositional(nn.Module):
    def __init__(self, ehr_features: int, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(ehr_features, dim)
        self.time_encoder = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, ehr: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        projected = self.projection(ehr)
        elapsed = torch.cumsum(delta, dim=1).unsqueeze(-1)
        positional = self.time_encoder(elapsed)
        return projected + positional
