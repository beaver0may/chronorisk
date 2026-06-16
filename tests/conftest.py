from __future__ import annotations

import pytest

from chronorisk.charts.passage import build_split
from chronorisk.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    RewardConfig,
    RqsConfig,
    TrainConfig,
)
from chronorisk.datum import TrajectoryBatch


def tiny_model() -> ModelConfig:
    return ModelConfig(
        ehr_features=10,
        bio_channels=2,
        bio_window=6,
        dim=16,
        heads=2,
        layers=2,
        conv_kernels=(3, 5),
        dropout=0.0,
    )


def tiny_data(seed: int = 3) -> DataConfig:
    return DataConfig(horizon=5, n_train=24, n_val=12, n_test=12, seed=seed)


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        name="tiny",
        model=tiny_model(),
        data=tiny_data(),
        reward=RewardConfig(),
        rqs=RqsConfig(),
        train=TrainConfig(
            batch_size=8,
            epochs=3,
            patience=5,
            pretrain_epochs=1,
            device="cpu",
        ),
    )


@pytest.fixture()
def config() -> ExperimentConfig:
    return tiny_config()


@pytest.fixture()
def batch(config: ExperimentConfig) -> TrajectoryBatch:
    return build_split(config.data.n_train, config.data.seed, config.model, config.data)
