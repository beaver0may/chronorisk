from __future__ import annotations

from chronorisk.config import load_experiment


def test_main_leg_loads_paper_defaults() -> None:
    config = load_experiment("main", [])
    assert config.name == "main"
    assert config.train.lr == 5e-4
    assert config.train.batch_size == 256
    assert config.train.gamma == 0.99
    assert config.train.cql_alpha == 0.1
    assert config.model.conv_kernels == (3, 5, 7)
    assert config.model.layers == 4
    assert len(config.train.seeds) == 15


def test_cli_overrides_are_coerced() -> None:
    config = load_experiment(
        "main",
        ["train.cql_alpha=0.5", "model.use_fusion=false", "data.dropped_channels=scifeat,wearable"],
    )
    assert config.train.cql_alpha == 0.5
    assert config.model.use_fusion is False
    assert config.data.dropped_channels == ("scifeat", "wearable")


def test_reward_weights_default_matches_manuscript() -> None:
    config = load_experiment("main", [])
    weights = (
        config.reward.w_pred,
        config.reward.w_trans,
        config.reward.w_cal,
        config.reward.w_safe,
    )
    assert weights == (0.3, 0.2, 0.2, 0.3)
