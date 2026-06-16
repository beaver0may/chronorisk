from __future__ import annotations

import torch
from torch import nn

from chronorisk.config import ModelConfig
from chronorisk.reckoning.timepe import TimeAwarePositional


class TemporalEhrEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.use_temporal = config.use_temporal
        self.embed = TimeAwarePositional(config.ehr_features, config.dim)
        if config.use_temporal:
            layer = nn.TransformerEncoderLayer(
                d_model=config.dim,
                nhead=config.heads,
                dim_feedforward=4 * config.dim,
                dropout=config.dropout,
                activation="gelu",
                batch_first=True,
            )
            self.encoder: nn.Module = nn.TransformerEncoder(layer, num_layers=config.layers)
        else:
            self.encoder = nn.Sequential(
                nn.Linear(config.dim, config.dim),
                nn.GELU(),
                nn.Linear(config.dim, config.dim),
            )
        self.reconstruction = nn.Linear(config.dim, config.ehr_features)

    def forward(self, ehr: torch.Tensor, delta: torch.Tensor) -> torch.Tensor:
        embedded = self.embed(ehr, delta)
        if self.use_temporal:
            horizon = embedded.shape[1]
            causal = torch.triu(
                torch.full((horizon, horizon), float("-inf"), device=embedded.device),
                diagonal=1,
            )
            return self.encoder(embedded, mask=causal)
        return self.encoder(embedded)

    def reconstruct_visits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.reconstruction(hidden)
