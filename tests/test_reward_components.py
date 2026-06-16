from __future__ import annotations

import torch

from chronorisk.config import RewardConfig
from chronorisk.setdrift.reward import composite_reward, prediction_reward, safety_reward


def test_under_triage_penalised_more_than_over_triage() -> None:
    outcome = torch.full((6,), 3)
    under = torch.full((6,), 1)
    over = outcome.clone()
    config = RewardConfig()
    safe_under = safety_reward(under, outcome, config.under_triage_lambda).mean().item()
    safe_match = safety_reward(over, outcome, config.under_triage_lambda).mean().item()
    assert safe_under < safe_match
    assert safe_match == 0.0


def test_prediction_reward_is_signed() -> None:
    action = torch.tensor([0, 1, 2, 3])
    outcome = torch.tensor([0, 1, 3, 3])
    reward = prediction_reward(action, outcome)
    assert reward.tolist() == [1.0, 1.0, -1.0, 1.0]


def test_composite_weighting_matches_components() -> None:
    config = RewardConfig()
    action = torch.tensor([1, 2])
    action_prev = torch.tensor([0, 2])
    outcome = torch.tensor([2, 2])
    total = composite_reward(action, action_prev, outcome, config)
    assert total.shape == action.shape
    assert torch.isfinite(total).all()
