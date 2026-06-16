from __future__ import annotations

import numpy as np

from chronorisk.config import RqsConfig


def under_triage_rate(assigned_tier: np.ndarray, outcome_tier: np.ndarray) -> float:
    return float(np.mean(assigned_tier < outcome_tier))


def _jensen_shannon(p: np.ndarray, q: np.ndarray) -> float:
    p = p / max(p.sum(), 1e-12)
    q = q / max(q.sum(), 1e-12)
    m = 0.5 * (p + q)

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        mask = a > 0
        return float(np.sum(a[mask] * np.log(a[mask] / b[mask])))

    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def transition_jsd(p_trans: np.ndarray, p_ref: np.ndarray) -> float:
    rows = p_trans.shape[0]
    divergences = [_jensen_shannon(p_trans[row], p_ref[row]) for row in range(rows)]
    return float(np.mean(divergences))


def recalibration_quality_score(
    auroc_value: float,
    ece_value: float,
    under_triage: float,
    jsd_value: float,
    config: RqsConfig,
) -> float:
    return float(
        auroc_value
        - config.lambda_cal * ece_value
        - config.lambda_under * under_triage
        - config.lambda_trans * jsd_value
    )
