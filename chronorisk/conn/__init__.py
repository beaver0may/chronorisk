from __future__ import annotations

import torch
from torch import nn

from chronorisk.bearings import BiosensorEncoder
from chronorisk.config import ModelConfig
from chronorisk.conn.cql import cql_loss, gather_action, soft_update
from chronorisk.conn.helm import HelmPolicy
from chronorisk.datum import ModelOutput, TrajectoryBatch
from chronorisk.fix import CrossAttentionFusion
from chronorisk.reckoning import TemporalEhrEncoder


class ChronoRisk(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.ehr_encoder = TemporalEhrEncoder(config)
        self.bio_encoder = BiosensorEncoder(config)
        self.fusion = CrossAttentionFusion(config)
        self.policy = HelmPolicy(config)

    def forward(self, batch: TrajectoryBatch) -> ModelOutput:
        ehr_hidden = self.ehr_encoder(batch.ehr, batch.delta)
        bio_hidden = self.bio_encoder(batch.bio, batch.bio_mask)
        state = self.fusion(ehr_hidden, bio_hidden)
        q_values = self.policy(state)
        reconstruction = self.ehr_encoder.reconstruct_visits(ehr_hidden)
        return ModelOutput(
            state=state,
            q_values=q_values,
            tier_logits=q_values,
            visit_reconstruction=reconstruction,
        )

    def tier_probabilities(self, batch: TrajectoryBatch) -> torch.Tensor:
        output = self.forward(batch)
        return torch.softmax(output.tier_logits, dim=-1)


def build_model(config: ModelConfig) -> ChronoRisk:
    return ChronoRisk(config)


__all__ = [
    "ChronoRisk",
    "build_model",
    "cql_loss",
    "gather_action",
    "soft_update",
]
