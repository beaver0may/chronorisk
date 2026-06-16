from __future__ import annotations

import torch
from torch import nn


def gather_action(q_values: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    return q_values.gather(-1, actions.unsqueeze(-1)).squeeze(-1)


def conservative_penalty(q_current: torch.Tensor, taken: torch.Tensor) -> torch.Tensor:
    return torch.logsumexp(q_current, dim=-1) - taken


def cql_loss(
    q_online: torch.Tensor,
    q_next_target: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    visit_mask: torch.Tensor,
    gamma: float,
    alpha: float,
) -> torch.Tensor:
    q_current = q_online[:, :-1]
    q_next = q_next_target[:, 1:].detach()
    chosen = actions[:, :-1]
    step_reward = rewards[:, :-1]
    taken = gather_action(q_current, chosen)
    target = step_reward + gamma * q_next.max(dim=-1).values
    bellman = (taken - target.detach()) ** 2
    penalty = conservative_penalty(q_current, taken)
    mask = (visit_mask[:, :-1] * visit_mask[:, 1:]).unsqueeze(-1)
    weighted = (alpha * penalty + 0.5 * bellman) * mask
    denominator = mask.expand_as(weighted).sum().clamp_min(1.0)
    return weighted.sum() / denominator


def soft_update(target: nn.Module, online: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(
            target.parameters(), online.parameters(), strict=False
        ):
            target_param.mul_(1.0 - tau).add_(tau * online_param)
        for target_buffer, online_buffer in zip(target.buffers(), online.buffers(), strict=False):
            target_buffer.copy_(online_buffer)
