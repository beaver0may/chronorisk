from __future__ import annotations

import torch
from torch import nn

from chronorisk.config import TrainConfig


def build_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=config.lr,
        weight_decay=config.weight_decay,
    )


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    config: TrainConfig,
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup = max(0, config.warmup)

    def factor(step: int) -> float:
        if warmup == 0:
            return 1.0
        return min(1.0, float(step + 1) / float(warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=factor)
