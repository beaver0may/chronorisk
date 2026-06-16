from __future__ import annotations

import torch

from chronorisk.charts.passage import build_split
from chronorisk.config import ExperimentConfig
from chronorisk.conn import build_model
from chronorisk.setdrift.objective import supervised_loss


def test_supervised_head_overfits_single_batch(config: ExperimentConfig) -> None:
    small = build_split(8, config.data.seed, config.model, config.data)
    model = build_model(config.model)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-3)
    first = float(supervised_loss(model, small).detach())
    for _ in range(200):
        optimizer.zero_grad()
        loss = supervised_loss(model, small)
        loss.backward()
        optimizer.step()
    final = float(supervised_loss(model, small).detach())
    assert final < 0.25 * first
