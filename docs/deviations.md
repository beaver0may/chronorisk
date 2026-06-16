# Deviations

Each entry links the manuscript location and states why the released implementation
differs from a literal transcription.

## D1 — Calibration reward surrogate
Ref: Methods §4.3 (Eq. 11, R_cal); §4.5 (Eq. 12).
The manuscript defines R_cal as the negative expected calibration error over a rolling
window of k = 50 patients. An offline reward must attach to each (state, action) pair
inside a trajectory, so R_cal is realised here as a per-element graded miscalibration
term (normalised tier distance between the assigned tier and the observed outcome tier).
The literal 10-bin ECE (Eq. 12) is implemented faithfully and used for evaluation and for
the Recalibration Quality Score; only the in-trajectory reward signal uses the surrogate.

## D2 — Risk-tier granularity
Ref: Methods §4.1 (A = {1,2,3,4}); Table 4 (tier-granularity sensitivity).
The default four-tier action space is implemented as a fixed enumeration. The Table 4
sensitivity rows for three- and five-tier configurations are documented but not provided
as separately runnable legs, because the tier count is a structural enumeration rather
than a numeric hyperparameter. The four-tier default matches the manuscript's primary and
best configuration.

## D3 — Datasets
Ref: Methods §4.4 (Datasets and cohort construction).
The seven source datasets are credentialed, application-gated, or controlled-tier and are
not redistributable. The released `charts` module generates seeded synthetic longitudinal
cohorts that reproduce the manuscript's trajectory structure (irregular intervals,
biosensor availability gaps, four-tier outcomes across five complications). Pointing the
loader at a licensed extract is the supported path to the reported numbers; the synthetic
legs validate the pipeline and metrics only.

## D4 — Unspecified model-shape constants
Ref: Results §2.6 (parameter budget); Methods §4.2 (architecture).
The manuscript states the parameter budget (12.8M; EHR encoder 8.2M, biosensor 2.1M, CQL
policy 0.8M) and "multi-head" attention without an explicit embedding dimension, head
count, or training precision. Reasoned defaults are adopted: embedding dimension 256
(consistent with the encoder budget at four layers), eight attention heads, and fp32 with
optional AMP. These are configurable in `legs/main.toml`.

## D5 — EpiCare benchmark
Ref: Methods §4.4 (EpiCare).
The RL-only benchmark is realised as a deterministic, seeded environment surrogate that
follows the reset/step contract described for the OpenAI Gym-conforming benchmark, without
importing the external EpiCare package, so the policy network can be exercised offline.

## D6 — Behaviour action for offline conservative Q-learning
Ref: Methods §4.1 (Assessment-MDP, Eq. 2-3, Definition 1).
The Assessment-MDP transition is action-independent: the assigned tier does not alter
patient biology. The released offline dataset therefore takes the standing (carried-
forward) tier as the behaviour action at each encounter and attaches the composite reward
relative to the observed outcome tier. This realises the sequential structure of tier
assignment without a separate logged behaviour policy, which the public registries do not
provide.
