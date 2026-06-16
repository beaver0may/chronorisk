from __future__ import annotations

import numpy as np


def expected_calibration_error(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = labels.size
    if total == 0:
        return 0.0
    error = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=False):
        if upper == edges[-1]:
            members = (probs >= lower) & (probs <= upper)
        else:
            members = (probs >= lower) & (probs < upper)
        count = int(members.sum())
        if count == 0:
            continue
        confidence = float(probs[members].mean())
        accuracy = float(labels[members].mean())
        error += (count / total) * abs(accuracy - confidence)
    return error


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    return float(np.mean((probs - labels.astype(np.float64)) ** 2))
