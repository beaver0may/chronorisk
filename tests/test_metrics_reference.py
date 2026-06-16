from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from chronorisk.soundingcheck.calibration import expected_calibration_error
from chronorisk.soundingcheck.discrimination import auprc, auroc


def _sample() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=400)
    scores = 0.5 * labels + rng.normal(0.0, 0.5, size=400)
    return scores, labels


def test_auroc_matches_sklearn() -> None:
    scores, labels = _sample()
    assert abs(auroc(scores, labels) - roc_auc_score(labels, scores)) < 1e-6


def test_auprc_matches_sklearn() -> None:
    scores, labels = _sample()
    assert abs(auprc(scores, labels) - average_precision_score(labels, scores)) < 1e-6


def test_ece_extreme_cases() -> None:
    confident_wrong = np.full(100, 0.9)
    labels = np.zeros(100)
    assert abs(expected_calibration_error(confident_wrong, labels) - 0.9) < 1e-6
    calibrated = np.concatenate([np.zeros(50), np.ones(50)])
    perfect_labels = np.concatenate([np.zeros(50), np.ones(50)])
    assert expected_calibration_error(calibrated, perfect_labels) < 1e-9
