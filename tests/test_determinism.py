from __future__ import annotations

import os
import tempfile

import torch

from chronorisk.charts.cohort import generate_cohort
from chronorisk.config import ExperimentConfig
from chronorisk.conn import build_model
from chronorisk.passageplan.almanac import set_seed
from chronorisk.passageplan.helmsman import build_optimizer
from chronorisk.passageplan.waypoint import load_checkpoint, save_checkpoint


def test_cohort_is_reproducible() -> None:
    first = generate_cohort(8, 5, 10, 2, 6, seed=11)
    second = generate_cohort(8, 5, 10, 2, 6, seed=11)
    assert (first.ehr == second.ehr).all()
    assert (first.tier_label == second.tier_label).all()


def test_seeded_model_outputs_match(config: ExperimentConfig, batch) -> None:
    set_seed(0)
    model_a = build_model(config.model)
    out_a = model_a(batch).q_values
    set_seed(0)
    model_b = build_model(config.model)
    out_b = model_b(batch).q_values
    assert torch.allclose(out_a, out_b)


def test_checkpoint_round_trip(config: ExperimentConfig) -> None:
    model = build_model(config.model)
    optimizer = build_optimizer(model, config.train)
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "ckpt.pt")
        save_checkpoint(path, model, optimizer, seed=99, epoch=3)
        payload = load_checkpoint(path)
    assert payload["seed"] == 99
    assert payload["epoch"] == 3
    restored = build_model(config.model)
    restored.load_state_dict(payload["model"])
    for left, right in zip(model.parameters(), restored.parameters(), strict=False):
        assert torch.allclose(left, right)
