from __future__ import annotations

from dataclasses import dataclass

import torch

from chronorisk.charts.passage import AssessmentMdp
from chronorisk.config import RqsConfig
from chronorisk.conn import ChronoRisk
from chronorisk.datum import RiskTier, TrajectoryBatch
from chronorisk.soundingcheck.calibration import brier_score, expected_calibration_error
from chronorisk.soundingcheck.discrimination import auprc, auroc
from chronorisk.soundingcheck.reclassification import category_free_nri
from chronorisk.soundingcheck.rqs import (
    recalibration_quality_score,
    transition_jsd,
    under_triage_rate,
)

_EVENT_TIER = int(RiskTier.HIGH)


@dataclass(frozen=True)
class EvaluationReport:
    auroc: float
    auprc: float
    cfnri: float
    ece: float
    brier: float
    under_triage: float
    transition_jsd: float
    rqs: float


def evaluate_model(
    model: ChronoRisk,
    batch: TrajectoryBatch,
    rqs_config: RqsConfig,
    discount: float,
) -> EvaluationReport:
    model.eval()
    with torch.no_grad():
        probs = model.tier_probabilities(batch)
    event_prob = probs[..., _EVENT_TIER:].sum(dim=-1)
    assigned = probs.argmax(dim=-1)
    outcome = batch.tier_label
    valid = (batch.visit_mask > 0.5).unsqueeze(-1).expand(outcome.shape).reshape(-1)
    selector = valid.bool().cpu().numpy()

    score = event_prob.reshape(-1).cpu().numpy()[selector]
    label = (outcome >= _EVENT_TIER).reshape(-1).cpu().numpy()[selector].astype("float64")
    reference = (
        (batch.prev_tier >= _EVENT_TIER).reshape(-1).cpu().numpy()[selector].astype("float64")
    )
    assigned_flat = assigned.reshape(-1).cpu().numpy()[selector]
    outcome_flat = outcome.reshape(-1).cpu().numpy()[selector]

    mdp = AssessmentMdp(discount=discount)
    p_trans = mdp.tier_transition_reference(assigned).cpu().numpy()
    p_ref = mdp.tier_transition_reference(outcome).cpu().numpy()

    au = auroc(score, label)
    ece = expected_calibration_error(score, label)
    under = under_triage_rate(assigned_flat, outcome_flat)
    jsd = transition_jsd(p_trans, p_ref)
    return EvaluationReport(
        auroc=au,
        auprc=auprc(score, label),
        cfnri=category_free_nri(score, reference, label),
        ece=ece,
        brier=brier_score(score, label),
        under_triage=under,
        transition_jsd=jsd,
        rqs=recalibration_quality_score(au, ece, under, jsd, rqs_config),
    )
