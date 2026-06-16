from __future__ import annotations

import numpy as np


def net_reclassification(
    new_scores: np.ndarray,
    old_scores: np.ndarray,
    labels: np.ndarray,
    thresholds: tuple[float, ...],
) -> float:
    bins = np.asarray(thresholds, dtype=np.float64)
    new_category = np.digitize(new_scores, bins)
    old_category = np.digitize(old_scores, bins)
    event = labels.astype(bool)
    nonevent = ~event
    up = new_category > old_category
    down = new_category < old_category

    def _safe(numerator: np.ndarray, denominator: np.ndarray) -> float:
        size = int(denominator.sum())
        if size == 0:
            return 0.0
        return float(numerator[denominator].sum() / size)

    nri_event = _safe(up, event) - _safe(down, event)
    nri_nonevent = _safe(down, nonevent) - _safe(up, nonevent)
    return float(nri_event + nri_nonevent)


def category_free_nri(new_scores: np.ndarray, old_scores: np.ndarray, labels: np.ndarray) -> float:
    event = labels.astype(bool)
    nonevent = ~event
    improvement = new_scores - old_scores

    def _proportion(direction: np.ndarray, mask: np.ndarray) -> float:
        size = int(mask.sum())
        if size == 0:
            return 0.0
        return float(direction[mask].sum() / size)

    up = improvement > 0
    down = improvement < 0
    event_term = _proportion(up, event) - _proportion(down, event)
    nonevent_term = _proportion(down, nonevent) - _proportion(up, nonevent)
    return float(event_term + nonevent_term)
