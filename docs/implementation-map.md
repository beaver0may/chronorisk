# Implementation Map

All paper provenance for the source tree lives here (the code carries no comments or
docstrings). Each numbered equation, algorithm, reported table, ablation, and metric is
mapped to the module that realises it.

## Equations

| paper loc | item | file | module symbol | notes |
|---|---|---|---|---|
| §4.1 Eq (2) | augmented state s_t = f_enc({x^EHR},{x^bio},Δt,a_{t-1}) | chronorisk/charts/passage.py | `assemble_state` | irregular Δt + previous tier a_{t-1} (uniform prior at t=0) |
| §4.1 Eq (3) | objective pi* = argmax E[Σ γ^t R]; CQL constraint Q̂≤Q_true | chronorisk/conn/cql.py | `ConservativeQLearner.objective` | γ=0.99; conservative lower bound |
| §4.2 Eq (4) | time-aware visit embedding e_τ = W_proj x + PE_time(Δt) | chronorisk/reckoning/timepe.py | `TimeAwarePositional` | learnable Δt function, NOT fixed sinusoid |
| §4.2 Eq (5) | h_T^EHR = TransformerEnc(e_{1:T}); L=4 | chronorisk/reckoning/transit.py | `TemporalEhrEncoder` | last-visit representation |
| §4.2 Eq (6) | h_T^bio = TemporalPool(Conv1D(x^bio)) | chronorisk/bearings/echosounder.py | `BiosensorEncoder` | dilated causal conv {3,5,7} + GRU-D mask gate m_τ |
| §4.2 Eq (7) | h̃^EHR = h^EHR + CrossAttn(Q=EHR,K=V=bio) | chronorisk/fix/crossbearing.py | `CrossAttentionFusion.ehr_branch` | bidirectional |
| §4.2 Eq (8) | h̃^bio = h^bio + CrossAttn(Q=bio,K=V=EHR) | chronorisk/fix/crossbearing.py | `CrossAttentionFusion.bio_branch` | scaled dot-product softmax(QK^T/√d)V |
| §4.2 Eq (9) | s_t = W_fuse[h̃^EHR ‖ h̃^bio] + b_fuse | chronorisk/fix/crossbearing.py | `CrossAttentionFusion.forward` | concat → state projection |
| §4.2 Eq (10) | CQL loss: α(E_s logΣ_a exp Q − E_D Q) + ½ TD² | chronorisk/conn/cql.py | `cql_loss` | α conservatism; OOD logsumexp penalty + Bellman |
| §4.3 Eq (11) | R = w1 R_pred + w2 R_trans + w3 R_cal + w4 R_safe | chronorisk/setdrift/reward.py | `composite_reward` | w=(0.3,0.2,0.2,0.3) |
| §4.3 Eq (11) | R_pred (+1 match / −1 else) | chronorisk/setdrift/reward.py | `prediction_reward` | tier vs observed severity |
| §4.3 Eq (11) | R_trans (penalise implausible jumps; reward smooth) | chronorisk/setdrift/reward.py | `transition_reward` | JSD-smoothed adjacency |
| §4.3 Eq (11) | R_cal (negative rolling ECE, k=50) | chronorisk/setdrift/reward.py | `calibration_reward` | rolling window |
| §4.3 Eq (11) | R_safe (asymmetric under-triage penalty −λ, λ>1) | chronorisk/setdrift/reward.py | `safety_reward` | missed high-risk costlier |
| §4.5 Eq (12) | ECE = Σ_b |B_b|/N · |acc−conf|, 10 bins | chronorisk/soundingcheck/calibration.py | `expected_calibration_error` | equal-width bins |
| §4.5 Eq (13) | RQS = AUROC − λ1 ECE − λ2 r_under − λ3 JSD(p_trans‖p_ref) | chronorisk/soundingcheck/rqs.py | `recalibration_quality_score` | λ1:λ2:λ3 = 0.5:1.0:0.3; held-out only |
| §4 (Methods) Eq (1) | NSBP Δ_RL(c) ≈ β(1 − AUROC_static(c)) | chronorisk/soundingcheck/nsbp.py | `fit_nsbp` | β=0.59, R²=0.61, n=12 |

## Architecture modules

| paper loc | module | file | symbol |
|---|---|---|---|
| §4.2 | temporal EHR encoder | chronorisk/reckoning/transit.py | `TemporalEhrEncoder` |
| §4.2 | biosensor signal encoder | chronorisk/bearings/echosounder.py | `BiosensorEncoder` |
| §4.2 | cross-attention fusion | chronorisk/fix/crossbearing.py | `CrossAttentionFusion` |
| §4.2 | CQL policy network | chronorisk/conn/helm.py | `HelmPolicy` (Q over 20 actions) |
| §4.2 | full assembled model | chronorisk/conn/__init__.py | `ChronoRisk` |
| §4.1 Def 1 | Assessment-MDP tuple (S,A,T,R,γ) | chronorisk/charts/passage.py | `AssessmentMdp`, `Transition` |
| §4.4 | EpiCare benchmark env surrogate | chronorisk/epicare/harbour.py | `EpiCareEnv` |

## Metrics & evaluation

| paper loc | metric | file | symbol |
|---|---|---|---|
| §4.5 (1) | AUROC | chronorisk/soundingcheck/discrimination.py | `auroc` |
| §4.5 (2) | AUPRC | chronorisk/soundingcheck/discrimination.py | `auprc` |
| §4.5 (3) | NRI / category-free NRI | chronorisk/soundingcheck/reclassification.py | `net_reclassification`, `category_free_nri` |
| §4.5 (4) Eq 12 | ECE | chronorisk/soundingcheck/calibration.py | `expected_calibration_error` |
| §4.5 (5) | Brier score | chronorisk/soundingcheck/calibration.py | `brier_score` |
| §4.5 | RQS | chronorisk/soundingcheck/rqs.py | `recalibration_quality_score` |
| §4.5 | WIS + FQE off-policy evaluation | chronorisk/soundingcheck/ope.py | `weighted_importance_sampling`, `fitted_q_evaluation` |
| §4.5 | paired bootstrap, DeLong, Holm-Bonferroni, BH-FDR | chronorisk/soundingcheck/discrimination.py | `paired_bootstrap`, `delong_test`, `holm_bonferroni`, `benjamini_hochberg` |

## Reported tables (configs/experiments)

| paper loc | table | covered by |
|---|---|---|
| Table 1 | main comparison, 18 baselines x 7 datasets | legs/main.toml + chronorisk/soundingcheck (per-dataset eval) |
| Table 2 | ablation (T1 component / T3 input-feature) | legs/ablation_*.toml |
| Table 3 | per-condition x SCI injury level AUROC | legs/supplementary_percondition.toml |
| Table 4 | hparam sensitivity (alpha / window / reward weights) | legs/ablation_alpha_*.toml, legs/ablation_window_*.toml, legs/ablation_reward_*.toml; tier-granularity rows documented in docs/deviations.md (D2) |
| Table 5 | computational cost | docs/repo-plan.md + README Vessel particulars |
| Table 6 | cross-domain transfer | legs/supplementary_transfer.toml |
| Table 7 | data-efficiency scaling (100..full N) | legs/supplementary_scaling.toml |
| Table 8 | pairwise component interaction (IR) | legs/ablation_pairwise_*.toml |
| Table 9 | Treatment-MDP vs Assessment-MDP | docs/implementation-map.md (formal contrast) + chronorisk/charts/passage.py docstring-free encoding (action-independent T_bio) |

## Ablations (Table 2) → component toggles

| ablation | config | mechanism |
|---|---|---|
| w/o RL module | legs/ablation_component_norl.toml | replace HelmPolicy with static cross-entropy scorer |
| w/o Temporal encoding | legs/ablation_component_notemporal.toml | bypass time-aware Transformer (point-in-time features) |
| w/o Biosensor fusion | legs/ablation_component_nofusion.toml | mask biosensor branch / cross-attention |
| w/o Recalibration | legs/ablation_component_norecal.toml | drop R_trans + discounted-return objective (per-encounter categorical) |
| Supervised + Composite | legs/ablation_component_supervised.toml | same 4-term loss, cross-entropy instead of CQL policy iteration |
| Conservative est. (alpha=0) | legs/ablation_alpha_000.toml | disable CQL conservatism |
| input-feature (T3) | legs/ablation_feature_{noscifeat,noautonomic,nowearable,nolabtrend,nomedhx}.toml | drop one input channel each |

## Tests

| kind | file |
|---|---|
| shape | tests/test_shapes.py |
| time-aware-PE irregular-interval invariant | tests/test_timepe_invariant.py |
| gradient flow (all params receive grad) | tests/test_gradient_flow.py |
| overfit single batch | tests/test_overfit_single_batch.py |
| metric correctness vs reference | tests/test_metrics_reference.py |
| CQL conservatism lower-bound property | tests/test_cql_conservatism.py |
| reward component (R_safe asymmetry) | tests/test_reward_components.py |
| RQS / JSD transition property | tests/test_rqs_transition.py |
| determinism + seed restore | tests/test_determinism.py |
| config TOML round-trip | tests/test_config_roundtrip.py |
| style guard (no comments/docstrings/forbidden phrases/emoji) | tests/test_style_guard.py |
| end-to-end training smoke (2 steps, loss decreases) | tests/test_training_smoke.py |
