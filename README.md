# PASSAGE PLAN — chronorisk

Vessel: ChronoRisk-AI offline conservative-RL recalibration engine
Waters: spinal-cord-injury chronic-disease risk tiers (npj Digital Medicine)
Plan form: a voyage runs in four legs — Appraisal, Planning, Execution, Monitoring —
the same order used here to take a clone-fresh checkout from install to reported numbers.

A risk tier assigned at discharge drifts as a patient recovers, develops new
complications, or decompensates. ChronoRisk-AI treats the standing tier as a position to
be re-fixed at every encounter: a time-aware Transformer dead-reckons the EHR trajectory,
a dilated-convolution echo-sounder reads the biosensor streams, a bidirectional
cross-attention takes a running fix from both, and a Conservative Q-Learning policy holds
the conn and decides the recalibrated tier over four levels and five chronic
complications.

---

## Leg 1 — Appraisal (what the voyage needs)

Engine room (one of three):

    pip install .
    # or
    conda env create -f environment.yml && conda activate chronorisk
    # or
    docker build -t chronorisk . && docker run --rm chronorisk chart

Python 3.11+ with PyTorch 2.9, NumPy 2.3, and plumbum. The science modules are written
against plain `torch.nn`; scikit-learn and scipy back the test-suite reference checks only.

Crew (package → station on board):

| station        | package              | role |
|----------------|----------------------|------|
| dead reckoning | `chronorisk.reckoning` | time-aware Transformer EHR encoder (Eq. 4-5) |
| echo sounder   | `chronorisk.bearings`  | dilated causal Conv1D biosensor encoder with GRU-D gate (Eq. 6) |
| running fix    | `chronorisk.fix`       | bidirectional cross-attention fusion (Eq. 7-9) |
| the conn       | `chronorisk.conn`      | CQL policy over 20 actions (Eq. 3, 10) |
| set and drift  | `chronorisk.setdrift`  | composite clinical reward (Eq. 11) |
| sounding check | `chronorisk.soundingcheck` | AUROC / AUPRC / NRI / ECE / Brier / RQS / OPE / NSBP |
| passage plan   | `chronorisk.passageplan` | seed control, atomic checkpoints, two-phase training |
| charts         | `chronorisk.charts`    | cohort + biosensor synthesis, Assessment-MDP assembly |
| harbour        | `chronorisk.epicare`   | RL-only benchmark environment surrogate |
| the bridge     | `chronorisk.bridge`    | command-line helm |

The shipped cohorts are produced by a seeded synthetic generator that matches the paper's
trajectory shapes (irregular visit intervals, biosensor availability gaps, four-tier
outcomes across five complications). The seven source datasets are access-controlled, so
no patient data travels with this repository; point the charts module at a licensed
extract to sail real waters.

## Leg 2 — Charts and publications (data)

| chart | version | access | waypoint |
|-------|---------|--------|----------|
| NSCISC | public-use de-identified | public | https://sites.uab.edu/nscisc/database/ |
| MIMIC-IV | v3.1 | credentialed | https://physionet.org/content/mimiciv/3.1/ |
| eICU-CRD | v2.0 | credentialed | https://physionet.org/content/eicu-crd/2.0/ |
| UK Biobank accelerometer | sub-study | application | https://www.ukbiobank.ac.uk/ |
| All of Us | v8 release | controlled | https://allofus.nih.gov/ |
| MIMIC-IV-ECG | v1.0 | open (ODbL v1.0) | https://physionet.org/content/mimic-iv-ecg/1.0/ |
| EpiCare | NeurIPS 2024 D&B | open | https://github.com/Grosenick-Lab-Cornell |

    python -m chronorisk.bridge chart

prints the same registry with licence and access flags.

## Leg 3 — Planning (legs and helm orders)

Each experiment is a TOML leg under `legs/`. `legs/main.toml` carries the manuscript
configuration; `legs/_smoke.toml` is the unit-test leg and is not for reporting. Helm
orders take `--leg <name>` and repeatable `--set section.key=value` overrides.

    python -m chronorisk.bridge train    --leg main --out runs/main.pt
    python -m chronorisk.bridge evaluate  --leg main --checkpoint runs/main.pt
    python -m chronorisk.bridge infer     --leg main --checkpoint runs/main.pt
    python -m chronorisk.bridge export    --leg main --checkpoint runs/main.pt --out runs/main.onnx

Ablation and supplementary legs cover the manuscript tables, for example:

    python -m chronorisk.bridge train --leg ablation_component_norl
    python -m chronorisk.bridge train --leg ablation_alpha_050
    python -m chronorisk.bridge train --leg supplementary_scaling --set data.n_train=500

## Leg 4 — Execution (waypoints and their charted positions)

The positions below are the manuscript values for the primary NSCISC validation and the
two temporal-clinical datasets. Reaching them requires the licensed cohorts in Leg 2; the
synthetic legs exercise the same pipeline and metrics without claiming these numbers.

| waypoint | helm order | charted position |
|----------|------------|------------------|
| NSCISC discrimination | `evaluate --leg main` | AUROC 0.918 +/- 0.011 |
| MIMIC-IV discrimination | `evaluate --leg supplementary_transfer --set data.dataset=mimic-iv` | AUROC 0.928 +/- 0.008 |
| eICU cross-site | `evaluate --leg main --set data.dataset=eicu-crd` | AUROC 0.912 +/- 0.010 |
| calibration | `evaluate --leg main` | ECE 0.041 |
| reclassification | `evaluate --leg main` | NRI 18.7% over Charlson reference |
| recalibration quality | `evaluate --leg main` | RQS 0.812 |
| RL-module ablation | `train --leg ablation_component_norl` | AUROC -4.0 pp vs full model |
| conservatism floor | `train --leg supplementary_scaling` | 92% retention at 500 trajectories |

## Monitoring (keep a continuous fix)

The framework's premise is that a position is never final; the same holds for the build.

    make lint     # ruff + black + isort
    make type     # mypy
    make test     # pytest

The suite spans shape checks, the time-aware-encoding irregular-interval property,
gradient flow across every parameter, single-batch overfitting, metric agreement with
scikit-learn, the CQL conservatism lower bound, reward asymmetry for under-triage, the
RQS transition-divergence property, determinism with seed restoration, config round-trip,
a style guard, and a two-phase training smoke that asserts the loss falls.

## Vessel particulars (compute)

| quantity | value |
|----------|-------|
| training hardware | 1x NVIDIA A100 40GB |
| training time | 45 GPU-hours per run (pre-train + fine-tune) |
| total study budget | ~2400 GPU-hours (15 seeds x 7 datasets x baselines + ablations) |
| inference latency | 38 ms per patient |
| parameters | 12.8M (EHR encoder 8.2M, biosensor 2.1M, CQL policy 0.8M) |
| precision | fp32 default; AMP optional |

Checkpoints are not distributed; the synthetic legs train in well under a minute on CPU
and stand in for a smoke of the full pipeline.

## Standing orders (ethics and limits)

Risk-tier recalibration is decision support, not autonomous triage; under-triage carries
asymmetric clinical cost and is penalised accordingly in the reward and scored separately
in the Recalibration Quality Score. Validation in the manuscript is retrospective and
off-policy; prospective, pre-registered evaluation is the next step before clinical use.
Injury-level specificity (cervical to cardiovascular, thoracolumbar to metabolic) is
claimed for spinal-cord-injury cohorts only and does not transfer to general populations.
Subgroup performance is reported across age, sex, and injury aetiology; the temporal
mechanism does not widen the demographic gaps present in static models.
