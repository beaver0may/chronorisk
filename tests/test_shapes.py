from __future__ import annotations

from chronorisk.config import ExperimentConfig
from chronorisk.conn import build_model
from chronorisk.datum import N_COMPLICATIONS, N_TIERS, TrajectoryBatch


def test_model_output_shapes(config: ExperimentConfig, batch: TrajectoryBatch) -> None:
    model = build_model(config.model)
    output = model(batch)
    horizon = config.data.horizon
    n = config.data.n_train
    assert output.state.shape == (n, horizon, config.model.dim)
    assert output.q_values.shape == (n, horizon, N_COMPLICATIONS, N_TIERS)
    assert output.visit_reconstruction.shape == (n, horizon, config.model.ehr_features)


def test_tier_probabilities_normalised(config: ExperimentConfig, batch: TrajectoryBatch) -> None:
    model = build_model(config.model)
    probs = model.tier_probabilities(batch)
    sums = probs.sum(dim=-1)
    assert ((sums - 1.0).abs() < 1e-5).all()
