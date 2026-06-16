from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from chronorisk.charts.soundings import synthesize_biosensor
from chronorisk.datum import N_COMPLICATIONS, N_TIERS

EHR_BLOCKS: dict[str, tuple[int, int]] = {
    "demographics": (0, 4),
    "labtrend": (4, 12),
    "autonomic": (12, 16),
    "medhx": (16, 22),
    "scifeat": (22, 28),
    "vitals": (28, 32),
}

_CHANNEL_ALIAS: dict[str, str] = {
    "nolabtrend": "labtrend",
    "noautonomic": "autonomic",
    "nomedhx": "medhx",
    "noscifeat": "scifeat",
    "nowearable": "wearable",
}


@dataclass(frozen=True)
class Cohort:
    ehr: np.ndarray
    delta: np.ndarray
    bio: np.ndarray
    bio_mask: np.ndarray
    prev_tier: np.ndarray
    tier_label: np.ndarray
    visit_mask: np.ndarray


def _normalize_dropped(dropped_channels: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(_CHANNEL_ALIAS.get(name, name) for name in dropped_channels)


def _tiers_from_severity(severity: np.ndarray) -> np.ndarray:
    return np.clip(np.round(severity), 0, N_TIERS - 1).astype(np.int64)


def generate_cohort(
    n_patients: int,
    horizon: int,
    ehr_features: int,
    bio_channels: int,
    bio_window: int,
    seed: int,
    dropped_channels: tuple[str, ...] = (),
) -> Cohort:
    rng = np.random.default_rng(seed)
    dropped = _normalize_dropped(dropped_channels)
    complications = N_COMPLICATIONS

    baseline = rng.uniform(0.2, 2.6, size=(n_patients, 1, complications))
    slope = rng.normal(0.15, 0.2, size=(n_patients, 1, complications))
    steps = np.arange(horizon).reshape(1, horizon, 1)
    noise = rng.normal(0.0, 0.25, size=(n_patients, horizon, complications))
    severity = np.clip(baseline + slope * steps + noise, 0.0, 3.0)

    tier_label = _tiers_from_severity(severity)
    prev_tier = np.empty_like(tier_label)
    prev_tier[:, 1:, :] = tier_label[:, :-1, :]
    prev_tier[:, 0, :] = rng.integers(0, N_TIERS, size=(n_patients, complications))

    delta = rng.uniform(0.5, 3.0, size=(n_patients, horizon)).astype(np.float32)
    delta[:, 0] = 0.0

    projection = rng.normal(0.0, 1.0, size=(complications, ehr_features)).astype(np.float32)
    severity_signal = severity.astype(np.float32) @ projection
    demographics = rng.normal(0.0, 1.0, size=(n_patients, 1, ehr_features)).astype(np.float32)
    ehr = severity_signal + demographics
    ehr = ehr + rng.normal(0.0, 0.3, size=(n_patients, horizon, ehr_features)).astype(np.float32)
    ehr = ehr.astype(np.float32)

    for name in dropped:
        if name in EHR_BLOCKS:
            start, stop = EHR_BLOCKS[name]
            ehr[:, :, start : min(stop, ehr_features)] = 0.0

    bio, bio_mask = synthesize_biosensor(severity, bio_channels, bio_window, rng, dropped)
    visit_mask = np.ones((n_patients, horizon), dtype=np.float32)

    return Cohort(
        ehr=ehr,
        delta=delta,
        bio=bio,
        bio_mask=bio_mask,
        prev_tier=prev_tier,
        tier_label=tier_label,
        visit_mask=visit_mask,
    )
