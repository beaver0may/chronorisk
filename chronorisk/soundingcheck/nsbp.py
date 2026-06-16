from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NsbpFit:
    beta: float
    intercept: float
    r_squared: float


def fit_nsbp(static_auroc: np.ndarray, rl_delta: np.ndarray) -> NsbpFit:
    headroom = 1.0 - static_auroc
    design = np.vstack([headroom, np.ones_like(headroom)]).T
    solution, _, _, _ = np.linalg.lstsq(design, rl_delta, rcond=None)
    beta, intercept = float(solution[0]), float(solution[1])
    prediction = design @ solution
    residual = float(np.sum((rl_delta - prediction) ** 2))
    total = float(np.sum((rl_delta - rl_delta.mean()) ** 2))
    r_squared = 1.0 - residual / total if total > 0 else 0.0
    return NsbpFit(beta=beta, intercept=intercept, r_squared=float(r_squared))
