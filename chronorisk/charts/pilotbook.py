from __future__ import annotations

from collections.abc import Iterator

import numpy as np

from chronorisk.datum import TrajectoryBatch


def _slice(batch: TrajectoryBatch, index: np.ndarray) -> TrajectoryBatch:
    selector = index.tolist()
    return TrajectoryBatch(
        ehr=batch.ehr[selector],
        delta=batch.delta[selector],
        bio=batch.bio[selector],
        bio_mask=batch.bio_mask[selector],
        prev_tier=batch.prev_tier[selector],
        tier_label=batch.tier_label[selector],
        visit_mask=batch.visit_mask[selector],
    )


def iterate_minibatches(
    batch: TrajectoryBatch,
    batch_size: int,
    seed: int,
    shuffle: bool = True,
) -> Iterator[TrajectoryBatch]:
    total = batch.batch_size
    order = np.arange(total)
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    step = max(1, batch_size)
    for start in range(0, total, step):
        index = order[start : start + step]
        if index.size == 0:
            continue
        yield _slice(batch, index)
