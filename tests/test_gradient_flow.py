from __future__ import annotations

import torch

from chronorisk.config import ExperimentConfig
from chronorisk.conn import build_model
from chronorisk.datum import TrajectoryBatch
from chronorisk.setdrift.objective import cql_objective, masked_visit_pretrain_loss, supervised_loss


def test_all_parameters_receive_gradient(config: ExperimentConfig, batch: TrajectoryBatch) -> None:
    online = build_model(config.model)
    target = build_model(config.model)
    target.load_state_dict(online.state_dict())
    generator = torch.Generator()
    generator.manual_seed(0)
    loss = cql_objective(online, target, batch, config.reward, config.train)
    loss = loss + 0.5 * supervised_loss(online, batch)
    loss = loss + masked_visit_pretrain_loss(online, batch, 0.5, generator)
    loss.backward()
    missing = [name for name, param in online.named_parameters() if param.grad is None]
    assert missing == []
