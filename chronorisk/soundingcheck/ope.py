from __future__ import annotations

import numpy as np


def weighted_importance_sampling(
    behaviour_probs: np.ndarray,
    target_probs: np.ndarray,
    rewards: np.ndarray,
    gamma: float,
) -> float:
    trajectories, horizon = rewards.shape
    ratios = np.clip(target_probs, 1e-8, None) / np.clip(behaviour_probs, 1e-8, None)
    cumulative = np.cumprod(ratios, axis=1)
    weights = cumulative[:, -1]
    discounts = gamma ** np.arange(horizon)
    returns = (rewards * discounts).sum(axis=1)
    normaliser = max(float(weights.sum()), 1e-8)
    return float((weights * returns).sum() / normaliser)


def fitted_q_evaluation(initial_q: np.ndarray) -> float:
    best = initial_q.max(axis=-1)
    return float(best.mean())
