from __future__ import annotations

import numpy as np

from chronorisk.config import RqsConfig
from chronorisk.soundingcheck.rqs import (
    recalibration_quality_score,
    transition_jsd,
    under_triage_rate,
)


def test_identical_transitions_have_zero_divergence() -> None:
    matrix = np.array([[0.7, 0.2, 0.1, 0.0], [0.1, 0.6, 0.2, 0.1]])
    assert transition_jsd(matrix, matrix) < 1e-9


def test_divergent_transitions_are_positive() -> None:
    a = np.array([[0.9, 0.1, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.1, 0.9]])
    assert transition_jsd(a, b) > 0.1


def test_under_triage_rate_counts_lower_assignments() -> None:
    assigned = np.array([0, 2, 3, 1])
    outcome = np.array([2, 2, 1, 3])
    assert abs(under_triage_rate(assigned, outcome) - 0.5) < 1e-9


def test_rqs_decreases_with_calibration_error() -> None:
    config = RqsConfig()
    good = recalibration_quality_score(0.9, 0.02, 0.05, 0.01, config)
    bad = recalibration_quality_score(0.9, 0.20, 0.05, 0.01, config)
    assert good > bad
