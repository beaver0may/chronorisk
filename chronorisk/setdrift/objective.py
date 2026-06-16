from __future__ import annotations

import torch
import torch.nn.functional as functional

from chronorisk.config import RewardConfig, TrainConfig
from chronorisk.conn import ChronoRisk
from chronorisk.conn.cql import cql_loss
from chronorisk.datum import N_TIERS, TrajectoryBatch
from chronorisk.setdrift.reward import compute_step_rewards


def masked_visit_pretrain_loss(
    model: ChronoRisk,
    batch: TrajectoryBatch,
    mask_ratio: float,
    generator: torch.Generator,
) -> torch.Tensor:
    output = model(batch)
    reconstruction = output.visit_reconstruction
    draw = torch.rand(batch.ehr.shape[:2], generator=generator, device=batch.ehr.device)
    masked = (draw < mask_ratio) & (batch.visit_mask > 0.5)
    selector = masked.unsqueeze(-1).expand_as(reconstruction)
    if selector.sum() == 0:
        return reconstruction.sum() * 0.0
    error = (reconstruction - batch.ehr) ** 2
    return error[selector].mean()


def supervised_loss(model: ChronoRisk, batch: TrajectoryBatch) -> torch.Tensor:
    output = model(batch)
    logits = output.tier_logits
    flat_logits = logits.reshape(-1, N_TIERS)
    flat_target = batch.tier_label.reshape(-1)
    weight = batch.visit_mask.unsqueeze(-1).expand(batch.tier_label.shape).reshape(-1)
    per_element = functional.cross_entropy(flat_logits, flat_target, reduction="none")
    return (per_element * weight).sum() / weight.sum().clamp_min(1.0)


def cql_objective(
    online: ChronoRisk,
    target: ChronoRisk,
    batch: TrajectoryBatch,
    reward_config: RewardConfig,
    train_config: TrainConfig,
) -> torch.Tensor:
    rewards = compute_step_rewards(batch, reward_config)
    q_online = online(batch).q_values
    with torch.no_grad():
        q_target = target(batch).q_values
    return cql_loss(
        q_online=q_online,
        q_next_target=q_target,
        actions=batch.prev_tier,
        rewards=rewards,
        visit_mask=batch.visit_mask,
        gamma=train_config.gamma,
        alpha=train_config.cql_alpha,
    )
