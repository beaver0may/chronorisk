# Repo Plan

## Directory tree

```
chronorisk/
  __init__.py            package surface (ChronoRisk, build_model, version)
  datum.py               typed tensor/state contracts, RiskTier, Complication, action index
  config.py              layered TOML -> frozen-dataclass loader (code default -> file -> CLI)
  charts/                data: cohort + biosensor synthesis + MDP trajectory assembly
    __init__.py
    registry.py          DatasetCard registry for the 7 datasets (name/version/license/url)
    cohort.py            synthetic longitudinal EHR cohort (visits, Δt, tier outcomes)
    soundings.py         biosensor stream synthesis (activity / HRV / ECG) + missing mask
    passage.py           AssessmentMdp, Transition, assemble_state, trajectory batching
    pilotbook.py         hand-written batch collation (no torch DataLoader)
  reckoning/             temporal EHR encoder
    __init__.py
    timepe.py            TimeAwarePositional (Eq 4)
    transit.py           TemporalEhrEncoder, masked-visit pretraining head (Eq 5)
  bearings/              biosensor encoder
    __init__.py
    echosounder.py       BiosensorEncoder dilated conv {3,5,7} + GRU-D gate (Eq 6)
  fix/                   cross-attention fusion
    __init__.py
    crossbearing.py      CrossAttentionFusion bidirectional (Eq 7-9)
  conn/                  CQL policy
    __init__.py          ChronoRisk full model assembly
    helm.py              HelmPolicy Q-network (20 actions)
    cql.py               cql_loss, ConservativeQLearner, soft target update (Eq 3,10)
  setdrift/              reward + objective
    __init__.py
    reward.py            composite_reward + 4 components (Eq 11)
    objective.py         training objective assembly (pretrain CE + CQL + reward shaping)
  soundingcheck/         metrics + evaluation
    __init__.py
    discrimination.py    auroc, auprc, paired_bootstrap, delong_test, holm_bonferroni, benjamini_hochberg
    calibration.py       expected_calibration_error (Eq 12), brier_score
    reclassification.py  net_reclassification, category_free_nri
    rqs.py               recalibration_quality_score (Eq 13), under_triage_rate, transition_jsd
    ope.py               weighted_importance_sampling, fitted_q_evaluation
    nsbp.py              fit_nsbp (Eq 1)
  passageplan/           training orchestration
    __init__.py
    almanac.py           set_seed, RNG snapshot/restore
    waypoint.py          atomic checkpoint (tmp + os.replace), seed persisted/restored
    helmsman.py          optimizer/scheduler/AMP/DDP-ready wiring
    pretrain.py          masked-visit pretraining loop (mask 15%)
    voyage.py            Trainer: pretrain -> fine-tune; early stop on val WIS
  epicare/
    __init__.py
    harbour.py           EpiCareEnv deterministic Gym-style surrogate
  bridge/                CLI (plumbum.cli)
    __init__.py
    __main__.py          ChronoRisk Application: chart / train / evaluate / infer / export
legs/                    TOML experiment configs
  main.toml  _smoke.toml
  ablation_component_*.toml  ablation_alpha_*.toml  ablation_window_*.toml
  ablation_tiers_*.toml  ablation_reward_*.toml  ablation_feature_*.toml  ablation_pairwise_*.toml
  supplementary_percondition.toml  supplementary_transfer.toml  supplementary_scaling.toml
tests/                   12 test kinds (see implementation-map)
docs/                    project-context.md  implementation-map.md  deviations.md
scripts/                 prepare_data.sh  launch_train.sh  launch_eval.sh
assets/
pyproject.toml  requirements.txt  environment.yml  Dockerfile  Makefile
.gitignore  .pre-commit-config.yaml  README.md  LICENSE
```

## Module responsibilities (one line each)

- `datum`: single source of truth for risk tiers (4), complications (5), 20-action index,
  and the typed batch/state structures passed between modules.
- `config`: read TOML into immutable dataclasses; CLI `key=value` overrides applied last.
- `charts`: produce paper-shaped longitudinal trajectories and biosensor streams entirely
  from a seeded generator; no real PHI download path is shipped.
- `reckoning` / `bearings` / `fix` / `conn`: the four manuscript modules.
- `setdrift`: the composite clinical reward and the combined training objective.
- `soundingcheck`: discrimination / calibration / reclassification / RQS / OPE / NSBP.
- `passageplan`: seed control, atomic checkpoints, optimizer wiring, two-phase training.
- `epicare`: the RL-only benchmark environment surrogate.
- `bridge`: the user-facing CLI verbs.

## Pinned dependencies

- python >= 3.11 (stdlib tomllib)
- torch == 2.9.*
- numpy == 2.3.*
- plumbum == 2.0.*
- scikit-learn == 1.7.* (reference metrics in tests only; library code is self-contained)
- scipy == 1.16.*
dev: ruff, black == 24.8.*, isort == 5.13.*, mypy, pytest == 9.0.*, pre-commit

## Expected test coverage

12 test files spanning shape, time-PE invariant, gradient flow, single-batch overfit,
metric correctness vs scikit-learn, CQL conservatism lower bound, reward asymmetry,
RQS/JSD transition property, determinism + seed restore, config round-trip, a style
guard (ast + tokenize), and a 2-step end-to-end training smoke that asserts the loss
decreases on `legs/_smoke.toml`.
