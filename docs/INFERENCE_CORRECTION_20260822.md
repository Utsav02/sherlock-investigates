# Inference correction specification — 2026-08-22

**Status:** frozen before implementation. Stage C is paused.

## Why this correction exists

Two historical analyses used uncertainty calculations that did not match their
data-generating structure:

1. V1 pooled the same eight prompts across many checkpoints from one training
   trajectory and treated all checkpoint×prompt outcomes as independent
   Bernoulli trials. Wilson intervals and Fisher's exact tests on those pooled
   totals are anti-conservative because prompts repeat and adjacent checkpoints
   share almost all of their training history.
2. Track A bootstrapped interrogators and human witnesses separately, then chose
   the wider interval. Participants appear in both roles, so this is not a valid
   treatment of the crossed/dyadic dependence.

The correction also separates three kinds of failure that the project previously
allowed to blur together:

- **scientific negative result:** a valid intervention does not change the target;
- **instrument or inference failure:** the measurement cannot support the claim;
- **planning failure:** implementation began before the construct, comparison,
  data source, or decision gate was adequate.

V1 contains all three. Raw Holmes prose not producing a verified investigative
policy is a useful negative observation. Calling checkpoint-pooled p-values
confirmatory was an inference failure. Training before verifying that prose
next-token prediction matched the intended policy, and building conversation
machinery before validating the commitment-gap instrument, were planning
failures. The methodological writeup must say this directly.

## V1 correction

### Available historical data

The committed dose-curve artifacts retain only one aggregate row per checkpoint:
`closure`, `n`, `truncated`, and mean token count. They do not retain which prompt
closed at which checkpoint or the generated text. Prompt-level covariance cannot
be reconstructed from these totals.

### Correct historical estimand

Historical checkpoint curves remain **descriptive trajectories from one trained
adapter path per condition**. Report checkpoint rates and ranges. Do not report:

- pooled Wilson intervals as though checkpoint×prompt rows were independent;
- Fisher p-values comparing pooled early and late checkpoints; or
- a causal verdict that steps, breadth, or rank is established from one training
  trajectory per condition.

The historical evidence can support: "closure was generally lower at later
checkpoints in these recorded trajectories." It cannot support a population-level
mechanism claim.

### Future-run requirement

Every generation must persist a prompt identifier, closure indicator, truncation
indicator, token count, sampling seed, and output hash. Analysis may use
prompt-blocked paired summaries within a trajectory. A causal training-factor
claim additionally requires independently trained seeds for each condition;
rerunning sampling from one adapter path is not a training replication.

## Track A correction

### Dependence structure

Each game is a dyad joining an interrogator and a human witness. A participant may
appear in either role across games. Games sharing either participant may be
dependent. Connected components control train/evaluation leakage and are also the
only clearly independent blocks in the released interaction graph.

### Primary and sensitivity intervals

For additive game-level outcomes (accuracy, Brier loss, dialogue-average Brier,
and dialogue-average log loss):

1. Report a **dyadic participant-cluster sandwich interval**. If `u_g` is the
   centered per-game contribution, the covariance meat is
   `sum_p (sum_{g incident to p} u_g)^2 - sum_g u_g^2`. This counts covariance
   between games sharing either endpoint while preserving a person's identity
   across roles. Apply a documented finite-sample correction and clip bounded
   metrics to their legal range.
2. Report a **connected-component bootstrap** sensitivity interval. It is likely
   conservative and has few clusters, but it makes the independence assumption
   explicit rather than calling a max-of-marginals construction conservative.

For non-additive outcomes such as AUROC and ECE, use the connected-component
bootstrap as the primary interval unless a separately tested influence-function
implementation is added. Do not synthesize a "participant" interval by selecting
the wider of two role-specific intervals.

These intervals condition on the already fitted cross-fold models. They quantify
variation across the observed participant/game structure, not the additional
variation that would arise from drawing a new training corpus and refitting the
whole algorithm. Any claim about algorithm-level generalization requires repeated
training samples or seeds; label the present intervals accordingly.

### Calibration

Calibration evaluation and calibration fitting remain distinct. Any new
calibrator must be nested:

- outer fold: untouched participants/components for evaluation;
- inner training portion: fit the representation head;
- inner calibration portion: fit Platt or isotonic calibration without seeing
  the outer fold;
- report raw-head and calibrated probabilities side by side.

The frozen final test split remains untouched.

## Hybrid deliverable decision

The owner selected a hybrid programme:

1. First package the methodological failure study, including poor sequencing and
   construct mismatch as causes rather than presenting every failure as a novel
   scientific finding.
2. Then continue toward real active investigation only through a bridge:
   provenance-locked out-of-corpus detector baseline, nested calibration, and a
   mandatory pre-registered real-replay transfer criterion for D0.

A positive synthetic Gate 2A result alone is not sufficient evidence to proceed
to a real-active claim.
