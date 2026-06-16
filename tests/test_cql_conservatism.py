from __future__ import annotations

import torch

from chronorisk.conn.cql import conservative_penalty, cql_loss, gather_action


def test_conservative_penalty_is_non_negative() -> None:
    torch.manual_seed(0)
    q_values = torch.randn(4, 5, 3, 4)
    actions = torch.randint(0, 4, (4, 5, 3))
    taken = gather_action(q_values, actions)
    penalty = conservative_penalty(q_values, taken)
    assert (penalty >= -1e-6).all()


def test_higher_alpha_raises_loss() -> None:
    torch.manual_seed(1)
    q_online = torch.randn(3, 4, 3, 4)
    q_target = torch.randn(3, 4, 3, 4)
    actions = torch.randint(0, 4, (3, 4, 3))
    rewards = torch.randn(3, 4, 3)
    visit_mask = torch.ones(3, 4)
    low = cql_loss(q_online, q_target, actions, rewards, visit_mask, gamma=0.99, alpha=0.0)
    high = cql_loss(q_online, q_target, actions, rewards, visit_mask, gamma=0.99, alpha=1.0)
    assert float(high) > float(low)
