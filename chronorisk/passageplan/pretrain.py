from __future__ import annotations

import logging

import torch

from chronorisk.charts.pilotbook import iterate_minibatches
from chronorisk.config import ExperimentConfig
from chronorisk.conn import ChronoRisk
from chronorisk.datum import TrajectoryBatch
from chronorisk.setdrift.objective import masked_visit_pretrain_loss

_LOGGER = logging.getLogger("chronorisk.pretrain")


def run_pretrain(
    model: ChronoRisk,
    train_batch: TrajectoryBatch,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
    generator: torch.Generator,
) -> list[float]:
    history: list[float] = []
    for epoch in range(config.train.pretrain_epochs):
        model.train()
        losses: list[float] = []
        for minibatch in iterate_minibatches(
            train_batch, config.train.batch_size, seed=config.data.seed + epoch
        ):
            optimizer.zero_grad()
            loss = masked_visit_pretrain_loss(model, minibatch, config.train.mask_ratio, generator)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        if losses:
            mean_loss = sum(losses) / len(losses)
            history.append(mean_loss)
            _LOGGER.info("pretrain epoch %d masked_visit_loss %.4f", epoch, mean_loss)
    return history
