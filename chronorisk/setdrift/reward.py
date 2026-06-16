from __future__ import annotations

import torch

from chronorisk.config import RewardConfig
from chronorisk.datum import N_TIERS, TrajectoryBatch

_SCALE: float = float(N_TIERS - 1)


def prediction_reward(action: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    match = action == outcome
    return torch.where(
        match,
        torch.ones_like(action, dtype=torch.float32),
        -torch.ones_like(action, dtype=torch.float32),
    )


def transition_reward(action: torch.Tensor, action_prev: torch.Tensor) -> torch.Tensor:
    jump = (action - action_prev).abs().to(torch.float32)
    excess = (jump - 1.0).clamp_min(0.0)
    return -excess / _SCALE


def calibration_reward(action: torch.Tensor, outcome: torch.Tensor) -> torch.Tensor:
    return -(action - outcome).abs().to(torch.float32) / _SCALE


def safety_reward(
    action: torch.Tensor, outcome: torch.Tensor, under_triage_lambda: float
) -> torch.Tensor:
    under_triage = (action < outcome).to(torch.float32)
    return -under_triage_lambda * under_triage


def composite_reward(
    action: torch.Tensor,
    action_prev: torch.Tensor,
    outcome: torch.Tensor,
    config: RewardConfig,
) -> torch.Tensor:
    return (
        config.w_pred * prediction_reward(action, outcome)
        + config.w_trans * transition_reward(action, action_prev)
        + config.w_cal * calibration_reward(action, outcome)
        + config.w_safe * safety_reward(action, outcome, config.under_triage_lambda)
    )


def compute_step_rewards(batch: TrajectoryBatch, config: RewardConfig) -> torch.Tensor:
    action = batch.prev_tier
    action_prev = torch.empty_like(action)
    action_prev[:, 1:, :] = action[:, :-1, :]
    action_prev[:, 0, :] = action[:, 0, :]
    return composite_reward(action, action_prev, batch.tier_label, config)
