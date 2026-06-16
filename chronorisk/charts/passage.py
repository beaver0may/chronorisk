from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from chronorisk.charts.cohort import Cohort, generate_cohort
from chronorisk.config import DataConfig, ModelConfig
from chronorisk.datum import N_COMPLICATIONS, N_TIERS, TrajectoryBatch


def _apply_temporal_window(visit_mask: np.ndarray, window: int) -> np.ndarray:
    if window <= 0:
        return visit_mask
    horizon = visit_mask.shape[1]
    masked = np.zeros_like(visit_mask)
    keep = min(window, horizon)
    masked[:, horizon - keep :] = visit_mask[:, horizon - keep :]
    return masked


def cohort_to_batch(cohort: Cohort, temporal_window: int = 0) -> TrajectoryBatch:
    visit_mask = _apply_temporal_window(cohort.visit_mask, temporal_window)
    return TrajectoryBatch(
        ehr=torch.from_numpy(cohort.ehr),
        delta=torch.from_numpy(cohort.delta),
        bio=torch.from_numpy(cohort.bio),
        bio_mask=torch.from_numpy(cohort.bio_mask),
        prev_tier=torch.from_numpy(cohort.prev_tier),
        tier_label=torch.from_numpy(cohort.tier_label),
        visit_mask=torch.from_numpy(visit_mask),
    )


def build_split(
    n_patients: int,
    seed: int,
    model: ModelConfig,
    data: DataConfig,
) -> TrajectoryBatch:
    cohort = generate_cohort(
        n_patients=n_patients,
        horizon=data.horizon,
        ehr_features=model.ehr_features,
        bio_channels=model.bio_channels,
        bio_window=model.bio_window,
        seed=seed,
        dropped_channels=data.dropped_channels,
    )
    return cohort_to_batch(cohort, data.temporal_window)


def make_splits(
    model: ModelConfig,
    data: DataConfig,
) -> tuple[TrajectoryBatch, TrajectoryBatch, TrajectoryBatch]:
    train = build_split(data.n_train, data.seed, model, data)
    val = build_split(data.n_val, data.seed + 1, model, data)
    test = build_split(data.n_test, data.seed + 2, model, data)
    return train, val, test


@dataclass(frozen=True)
class AssessmentMdp:
    discount: float

    def tier_transition_reference(self, tier_label: torch.Tensor) -> torch.Tensor:
        current = tier_label[:, :-1, :].reshape(-1)
        nxt = tier_label[:, 1:, :].reshape(-1)
        table = torch.zeros(N_TIERS, N_TIERS)
        for src, dst in zip(current.tolist(), nxt.tolist(), strict=False):
            table[int(src), int(dst)] += 1.0
        table = table + 1e-6
        return table / table.sum(dim=1, keepdim=True)

    def is_action_independent(self) -> bool:
        return True

    @property
    def n_actions(self) -> int:
        return N_COMPLICATIONS * N_TIERS
