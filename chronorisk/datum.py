from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


class RiskTier(IntEnum):
    LOW = 0
    MODERATE = 1
    HIGH = 2
    VERY_HIGH = 3


class Complication(IntEnum):
    CARDIOVASCULAR = 0
    METABOLIC = 1
    URINARY_TRACT = 2
    PRESSURE_INJURY = 3
    OSTEOPOROSIS = 4


N_TIERS: int = len(RiskTier)
N_COMPLICATIONS: int = len(Complication)
N_ACTIONS: int = N_TIERS * N_COMPLICATIONS


def action_index(complication: int, tier: int) -> int:
    return complication * N_TIERS + tier


def split_action(index: int) -> tuple[int, int]:
    return index // N_TIERS, index % N_TIERS


@dataclass(frozen=True)
class TrajectoryBatch:
    ehr: torch.Tensor
    delta: torch.Tensor
    bio: torch.Tensor
    bio_mask: torch.Tensor
    prev_tier: torch.Tensor
    tier_label: torch.Tensor
    visit_mask: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.ehr.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.ehr.shape[1])

    def to(self, device: torch.device) -> TrajectoryBatch:
        return TrajectoryBatch(
            ehr=self.ehr.to(device),
            delta=self.delta.to(device),
            bio=self.bio.to(device),
            bio_mask=self.bio_mask.to(device),
            prev_tier=self.prev_tier.to(device),
            tier_label=self.tier_label.to(device),
            visit_mask=self.visit_mask.to(device),
        )


@dataclass(frozen=True)
class ModelOutput:
    state: torch.Tensor
    q_values: torch.Tensor
    tier_logits: torch.Tensor
    visit_reconstruction: torch.Tensor
