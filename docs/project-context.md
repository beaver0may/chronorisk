# Project Context

project_name       : chronorisk                                [HIGH]
domain             : spinal-cord-injury chronic-disease risk-tier recalibration
                     via offline conservative RL over fused biosensor + EHR streams   [HIGH]
framework          : PyTorch 2.x + plain torch.nn (hand-written CQL, time-aware
                     Transformer encoder, dilated Conv1D biosensor encoder,
                     bidirectional cross-attention)                                    [HIGH]
venue              : npj Digital Medicine                                              [HIGH]
primary_datasets   : 7 datasets (see §6)                                              [HIGH]
compute_target     : 1x NVIDIA A100 40GB; 45 GPU-h / run; ~2400 GPU-h total;
                     38 ms inference / patient; 12.8M parameters                       [HIGH]
hparams_reference  : Methods §4.5 + Table 4 + Fig. 6/7                                 [HIGH]
supp_path          : none

NEEDS_USER_DECISION: 0 unresolved (3 model-shape fields carry reasoned defaults below)
Status             : resolved.

---

## 1. project_name

`chronorisk` — snake_case package name. Content words from the title model name
"ChronoRisk-AI"; the "-AI" stopword is dropped.
Location: Title (p. 1); model name used throughout §2-§4. Confidence: HIGH.

## 2. supp_path

`none`. Globbing sibling locations (`paper/*`, `*supp*`, `*supplement*`, `*_si.*`,
`appendix*`) finds only other unrelated manuscript PDFs in `paper/`; this manuscript
embeds its supplementary tables (S1-S34) by reference only — no separate SI file exists
on disk. All supplementary references resolve to in-text table numbers.
Confidence: HIGH.

## 3. domain

Computational chronic-disease risk stratification for spinal cord injury (SCI):
continuous risk-tier recalibration formulated as an offline conservative reinforcement
learning problem over irregularly sampled longitudinal EHR fused with continuous
biosensor streams.
Location: Abstract (p. 1); Results §2.1; Methods §4.1-§4.3. Confidence: HIGH.

## 4. framework

PyTorch 2.x with plain `torch.nn`. The Methods describe a Transformer temporal encoder
with time-aware positional encoding (Eq. 4-5), a 1D dilated causal convolutional
biosensor encoder with GRU-D-style masking (Eq. 6), bidirectional scaled-dot-product
cross-attention (Eq. 7-9), and a Conservative Q-Learning policy network (Eq. 3, 10).
No third-party RL or graph library is named; CQL, the encoders, and the attention are
implemented directly. The EpiCare benchmark conforms to OpenAI Gym; the released
environment surrogate follows the same step/reset contract without external deps.
Location: §4.2 operator/class fingerprints; §4.5 training. Confidence: HIGH.

## 5. venue

npj Digital Medicine. Structured abstract, npj-style line numbering, separate Data
Availability / Code Availability headings, and the manuscript's own framing
("deployment model established by other npj Digital Medicine RL frameworks", §3).
Location: front matter; §3. Confidence: HIGH.

## 6. primary_datasets

All access URLs below were resolved live; every link is reachable (UK Biobank returns
403 to automated fetchers but the canonical site is live). No link was discarded.

1. NSCISC — National Spinal Cord Injury Statistical Center (primary SCI cohort)
   - version: public-use de-identified release (pre-September-2021 records; registry as of Feb 2026 = 55,715)
   - license: public de-identified data, NSCISC public-use portal terms
   - url: https://sites.uab.edu/nscisc/database/
   - n = 55,715; chronic-complication task split 39,001 / 5,572 / 11,142 (patient-level 70/10/20)

2. MIMIC-IV — temporal modeling validation
   - version: v3.1 (released 2024-10-11)
   - license: PhysioNet Credentialed Health Data License 1.5.0 (credentialed access)
   - url: https://physionet.org/content/mimiciv/3.1/
   - 73,181 ICU-stay subset; split 51,227 / 7,318 / 14,636 (patient-level, dedup vs MIMIC-IV-ECG)

3. eICU-CRD — cross-site generalization
   - version: v2.0 (published 2019-04-15)
   - license: PhysioNet Credentialed Health Data License 1.5.0 (credentialed access)
   - url: https://physionet.org/content/eicu-crd/2.0/
   - 200,859 encounters / 208 hospitals; split 140,601 / 20,086 / 40,172

4. UK Biobank accelerometer sub-study — biosensor validation
   - version: 7-day wrist accelerometer (Axivity AX3, 100 Hz)
   - license: access via UK Biobank Access Management System application approval
   - url: https://www.ukbiobank.ac.uk/
   - 103,712 participants; split 72,598 / 10,371 / 20,743

5. All of Us Research Program — multimodal fusion at scale
   - version: v8 data release (2025-02)
   - license: controlled access via All of Us Researcher Workbench
   - url: https://allofus.nih.gov/
   - 633,000+ adults with Fitbit data; patient-level demographic-stratified split

6. MIMIC-IV-ECG — ECG-EHR fusion
   - version: v1.0 (published 2023-09-15)
   - license: Open Data Commons Open Database License v1.0 (open access)
   - url: https://physionet.org/content/mimic-iv-ecg/1.0/
   - ~800,000 diagnostic ECGs / ~160,000 patients; 12,847 overlapping ids removed vs MIMIC-IV

7. EpiCare — RL benchmark (simulated, isolates the policy network)
   - version: NeurIPS 2024 Datasets & Benchmarks Track, OpenAI Gym-conforming
   - license: open-source, associated code repository
   - url: https://github.com/Grosenick-Lab-Cornell (EpiCare project; exact repo confirmed at build time)
   - 8 simulated clinical environments

Aggregate test set N_test = 245,564; primary NSCISC N_test = 11,142; total N > 1.2M.

## 7. compute_target

1x NVIDIA A100 (40 GB). 45 GPU-hours per run (pre-training on MIMIC-IV + fine-tuning
on NSCISC). Total budget across 15 seeds x 7 datasets x 18 baselines + ablations
~2400 GPU-hours. Inference latency 38 ms / patient. 12.8M parameters
(temporal EHR encoder 8.2M, biosensor encoder 2.1M, CQL policy 0.8M; remainder fusion).
world_size = 1 (single A100 implied by per-run hours and single-GPU inference).
Location: §2.6; Table 5; §4.7 Reproducibility statement. Confidence: HIGH.

## 8. hparams_reference

Methods §4.5 ("RL training") plus Table 4 (sensitivity) and Fig. 6/7 strips:
- optimizer Adam, lr 5e-4, batch_size 256, epochs 200, early-stop patience 20 epochs
- discount gamma 0.99, target-network soft update tau 0.005, CQL conservatism alpha 0.1
- CQL alpha grid {0.001, 0.01, 0.1, 0.5, 1.0}
- composite reward weights w1..w4 = 0.3, 0.2, 0.2, 0.3
- RQS penalty weights lambda2 : lambda1 : lambda3 = 1.0 : 0.5 : 0.3
- Transformer L = 4 encoder layers; dilated conv kernels {3, 5, 7}
- action space A = {1,2,3,4} (4 risk tiers) x 5 chronic complications = 20 actions
- 15 seeds: 42, 123, 256, 389, 512, 678, 741, 853, 927, 1024, 1111, 1337, 1500, 1776, 2025
- ECE 10 equal-width bins; R_cal rolling window k = 50
Location: §4.5; Table 4; Fig. 6/7. Confidence: HIGH.

## 9. extra_signals

- Algorithm/equation boxes: Eq. (1)-(13) numbered; Assessment-MDP Definition 1; no
  separate pseudocode boxes.
- Reported tables to mirror: Table 1 (main, 18 baselines x 7 datasets), Table 2
  (ablation), Table 3 (per-condition x SCI level), Table 4 (hparam sensitivity),
  Table 5 (compute), Table 6 (cross-domain transfer), Table 7 (data-efficiency scaling),
  Table 8 (pairwise interaction), Table 9 (Treatment-MDP vs Assessment-MDP).
- SI-only experiments to mirror as supplementary configs: cfNRI (S1), DeLong/ROC (S13),
  robustness under missing modalities + label noise (S20/S38), seed stability (S20),
  decision-curve (S36), pre/post-2020 temporal shift (S19), reward-lambda sensitivity (S34).
- No proprietary tokenizer. No released checkpoints (paper states code on publication).
- Three model-shape fields not given an explicit number in the manuscript; reasoned
  defaults adopted (recorded so the default config is faithful to the param budget):
    * embedding dim d = 256 (derived: 8.2M EHR-encoder params at L=4 ~ d in [192, 320];
      256 lands the budget) [MED]
    * attention heads = 8 (standard for d=256; "multi-head" unspecified) [MED]
    * precision = fp32, AMP optional/off by default (unstated; consistent with
      45 GPU-h on a single 40GB A100) [MED]
- Code-availability statement (verbatim, kept here only, NOT in README):
  "Code for reproducing the experiments will be made available at Github upon publication."
- Ethics/fairness: demographic subgroup parity reported (§2.4); no restrictive fairness
  demands beyond equitable-performance reporting.
