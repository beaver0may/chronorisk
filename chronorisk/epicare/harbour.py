from __future__ import annotations

from typing import Any

import numpy as np

from chronorisk.datum import N_TIERS


class EpiCareEnv:
    def __init__(self, n_features: int = 8, horizon: int = 10, seed: int = 0) -> None:
        self.n_features = n_features
        self.horizon = horizon
        self.n_actions = N_TIERS
        self._rng = np.random.default_rng(seed)
        self._projection = self._rng.normal(0.0, 1.0, size=(n_features,))
        self._severity = 0.0
        self._step = 0

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._severity = float(self._rng.uniform(0.0, float(N_TIERS - 1)))
        self._step = 0
        return self._observe()

    def _observe(self) -> np.ndarray:
        noise = self._rng.normal(0.0, 0.1, size=(self.n_features,))
        return (self._projection * self._severity + noise).astype(np.float32)

    def _hidden_tier(self) -> int:
        return int(np.clip(round(self._severity), 0, N_TIERS - 1))

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        target = self._hidden_tier()
        match = 1.0 if action == target else -1.0
        under_triage = -1.0 if action < target else 0.0
        reward = float(match + 0.5 * under_triage)
        drift = float(self._rng.normal(0.1, 0.2))
        self._severity = float(np.clip(self._severity + drift, 0.0, float(N_TIERS - 1)))
        self._step += 1
        terminated = self._step >= self.horizon
        observation = self._observe()
        return observation, reward, terminated, False, {"hidden_tier": target}
