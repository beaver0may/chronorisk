from __future__ import annotations

from chronorisk.config import load_experiment
from chronorisk.passageplan.voyage import Trainer


def test_two_phase_training_reduces_loss() -> None:
    config = load_experiment("_smoke", [])
    _, result = Trainer().fit(config)
    assert len(result.finetune_history) >= 2
    assert result.finetune_history[-1] < result.finetune_history[0]
    assert result.epochs_run >= 1
