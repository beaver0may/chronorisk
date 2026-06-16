from __future__ import annotations

import numpy as np
from scipy.stats import norm, rankdata


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    ranks = rankdata(scores)
    statistic = ranks[positive].sum() - n_pos * (n_pos + 1) / 2.0
    return float(statistic / (n_pos * n_neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(-scores)
    ordered = labels[order].astype(np.float64)
    cumulative = np.cumsum(ordered)
    precision = cumulative / np.arange(1, ordered.size + 1)
    total = max(float(ordered.sum()), 1.0)
    recall = cumulative / total
    previous = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - previous) * precision))


def paired_bootstrap(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    labels: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    size = labels.size
    differences = np.empty(resamples)
    for index in range(resamples):
        pick = rng.integers(0, size, size=size)
        differences[index] = auroc(scores_a[pick], labels[pick]) - auroc(
            scores_b[pick], labels[pick]
        )
    lower = float(np.quantile(differences, 0.025))
    upper = float(np.quantile(differences, 0.975))
    return float(differences.mean()), lower, upper


def _midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    size = values.size
    ranks = np.zeros(size)
    index = 0
    while index < size:
        run = index
        while run < size and sorted_values[run] == sorted_values[index]:
            run += 1
        ranks[index:run] = 0.5 * (index + run - 1) + 1.0
        index = run
    out = np.empty(size)
    out[order] = ranks
    return out


def delong_test(
    scores_a: np.ndarray, scores_b: np.ndarray, labels: np.ndarray
) -> tuple[float, float]:
    positive = labels.astype(bool)
    pos_a, neg_a = scores_a[positive], scores_a[~positive]
    pos_b, neg_b = scores_b[positive], scores_b[~positive]
    m, n = pos_a.size, neg_a.size
    if m == 0 or n == 0:
        return 0.0, 1.0
    auc = np.array([auroc(scores_a, labels), auroc(scores_b, labels)])
    v10 = np.empty((2, m))
    v01 = np.empty((2, n))
    for row, (pos, neg) in enumerate(((pos_a, neg_a), (pos_b, neg_b))):
        tx = _midrank(pos)
        ty = _midrank(neg)
        txy = _midrank(np.concatenate([pos, neg]))
        v10[row] = (txy[:m] - tx) / n
        v01[row] = 1.0 - (txy[m:] - ty) / m
    s10 = np.cov(v10)
    s01 = np.cov(v01)
    covariance = s10 / m + s01 / n
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    if variance <= 0.0:
        return 0.0, 1.0
    z = float((auc[0] - auc[1]) / np.sqrt(variance))
    p = float(2.0 * (1.0 - norm.cdf(abs(z))))
    return z, p


def holm_bonferroni(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    total = pvalues.size
    adjusted = np.empty(total)
    running = 0.0
    for rank, index in enumerate(order):
        value = (total - rank) * pvalues[index]
        running = max(running, value)
        adjusted[index] = min(running, 1.0)
    return adjusted


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    order = np.argsort(pvalues)
    total = pvalues.size
    adjusted = np.empty(total)
    running = 1.0
    for position in range(total - 1, -1, -1):
        index = order[position]
        value = pvalues[index] * total / (position + 1)
        running = min(running, value)
        adjusted[index] = min(running, 1.0)
    return adjusted
