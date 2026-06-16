from __future__ import annotations

import numpy as np


def synthesize_biosensor(
    severity: np.ndarray,
    channels: int,
    window: int,
    rng: np.random.Generator,
    dropped_channels: tuple[str, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    n_patients, horizon, _ = severity.shape
    drive = 0.6 * severity[:, :, 0] + 0.4 * severity[:, :, 1]
    grid = np.linspace(0.0, 1.0, window).reshape(1, 1, 1, window)
    base = drive[:, :, None, None]
    channel_phase = rng.uniform(0.0, 2.0 * np.pi, size=(1, 1, channels, 1))
    oscillation = np.sin(2.0 * np.pi * (1.0 + base) * grid + channel_phase)
    amplitude = 0.5 + 0.3 * base
    signal = oscillation * amplitude
    signal = signal + rng.normal(0.0, 0.1, size=(n_patients, horizon, channels, window))
    bio = signal.astype(np.float32)
    bio_mask = (rng.uniform(0.0, 1.0, size=(n_patients, horizon)) < 0.7).astype(np.float32)
    if "wearable" in dropped_channels:
        bio = np.zeros_like(bio)
        bio_mask = np.zeros((n_patients, horizon), dtype=np.float32)
    return bio, bio_mask
