from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import torch

from chronorisk.charts.passage import make_splits
from chronorisk.charts.pilotbook import iterate_minibatches
from chronorisk.config import ExperimentConfig
from chronorisk.conn import ChronoRisk, build_model
from chronorisk.conn.cql import soft_update
from chronorisk.datum import TrajectoryBatch
from chronorisk.passageplan.almanac import set_seed
from chronorisk.passageplan.helmsman import build_optimizer
from chronorisk.passageplan.pretrain import run_pretrain
from chronorisk.setdrift.objective import cql_objective, supervised_loss
from chronorisk.soundingcheck.evaluate import EvaluationReport, evaluate_model

_LOGGER = logging.getLogger("chronorisk.voyage")


@dataclass(frozen=True)
class TrainingResult:
    finetune_history: list[float]
    pretrain_history: list[float]
    best_rqs: float
    epochs_run: int
    validation: EvaluationReport


def _finetune_step(
    online: ChronoRisk,
    target: ChronoRisk,
    batch: TrajectoryBatch,
    config: ExperimentConfig,
    optimizer: torch.optim.Optimizer,
) -> float:
    optimizer.zero_grad()
    if online.config.use_rl:
        loss = cql_objective(online, target, batch, config.reward, config.train)
        loss = loss + 0.5 * supervised_loss(online, batch)
    else:
        loss = supervised_loss(online, batch)
    loss.backward()
    if config.train.grad_clip > 0.0:
        torch.nn.utils.clip_grad_norm_(online.parameters(), config.train.grad_clip)
    optimizer.step()
    if online.config.use_rl:
        soft_update(target, online, config.train.tau)
    return float(loss.detach())


class Trainer:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed

    def fit(self, config: ExperimentConfig) -> tuple[ChronoRisk, TrainingResult]:
        seed = self.seed if self.seed is not None else config.train.seeds[0]
        set_seed(seed)
        device = torch.device(config.train.device)
        train, val, _ = make_splits(config.model, config.data)
        train = train.to(device)
        val = val.to(device)

        online = build_model(config.model).to(device)
        target = build_model(config.model).to(device)
        target.load_state_dict(online.state_dict())
        optimizer = build_optimizer(online, config.train)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)

        pretrain_history = run_pretrain(online, train, config, optimizer, generator)
        target.load_state_dict(online.state_dict())

        history: list[float] = []
        best_rqs = float("-inf")
        best_state = copy.deepcopy(online.state_dict())
        best_report = evaluate_model(online, val, config.rqs, config.train.gamma)
        patience = 0
        epochs_run = 0
        for epoch in range(config.train.epochs):
            online.train()
            losses: list[float] = []
            for minibatch in iterate_minibatches(
                train, config.train.batch_size, seed=seed + epoch + 1
            ):
                losses.append(_finetune_step(online, target, minibatch, config, optimizer))
            epochs_run += 1
            if losses:
                history.append(sum(losses) / len(losses))
            report = evaluate_model(online, val, config.rqs, config.train.gamma)
            _LOGGER.info("epoch %d rqs %.4f auroc %.4f", epoch, report.rqs, report.auroc)
            if report.rqs > best_rqs + 1e-6:
                best_rqs = report.rqs
                best_state = copy.deepcopy(online.state_dict())
                best_report = report
                patience = 0
            else:
                patience += 1
            if patience >= config.train.patience:
                break

        online.load_state_dict(best_state)
        result = TrainingResult(
            finetune_history=history,
            pretrain_history=pretrain_history,
            best_rqs=best_rqs,
            epochs_run=epochs_run,
            validation=best_report,
        )
        return online, result
