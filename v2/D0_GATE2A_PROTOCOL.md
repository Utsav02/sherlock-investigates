# D0 Gate 2A protocol (frozen before implementation and run)

**Frozen:** 2026-08-22

**Scope:** exact synthetic-active mechanics only; no model inference, SFT, GPU,
human collection, or Track A test access.

## Purpose

Gate 2A asks whether question selection itself can improve identity inference in
a controlled environment where every response likelihood is known. It is a
mechanics sanity check, not evidence that the simulator resembles real people or
current AI systems.

The primary policy is exact sequential Bayesian experimental design (BED): at
each turn choose the unused question with maximum expected information gain
(EIG) under the current posterior and the true frozen likelihood table. This
implements the central BED-LLM objective without an LLM likelihood estimator,
because D0 makes that approximation unnecessary.

UoT is retained as a secondary heuristic. It draws one simulated response under
each latent type per candidate question, treats those samples as a deterministic
partition, and chooses the candidate with maximum simulated response entropy.
It is explicitly “UoT-style,” not a reproduction of the paper's LLM-generated
multi-step tree.

## Exact generative model

The latent type is binary: `human` or `ai`, with prior 0.5. An episode belongs to
one scenario family and exposes twelve question strategies. Each response has
one of three mutually exclusive categories: `human_cue`, `neutral`, or `ai_cue`.

For family `f` and question `q`, the frozen config supplies signed
discrimination `d[f,q]` and neutral mass `n[f]`. Conditional probabilities are:

```text
P(neutral | human,q,f) = P(neutral | ai,q,f) = n
P(human_cue | human,q,f) = (1-n)(1+d)/2
P(ai_cue    | human,q,f) = (1-n)(1-d)/2
P(human_cue | ai,q,f)    = (1-n)(1-d)/2
P(ai_cue    | ai,q,f)    = (1-n)(1+d)/2
```

Negative `d` represents an inverted cue. No probability is zero; answers are
partially informative and a later answer can recover from an early misleading
one. Given the type, the complete response schedule contains one independent
draw per question. Every policy in a matched episode sees the same pre-drawn
answer if it asks that question (common random numbers).

The emitted text is selected uniformly from the category's rendering bank, so
its probability is the category probability divided by the bank size. Text does
not determine the authoritative update: the simulator records the response
category and exact likelihood, and Bayes' rule owns the ledger probability.

## Question bank and provenance

The twelve questions map to strategy codes in the released Jones & Bergen
`tt_strategyclass.csv`: daily activities (`1b`), opinions (`5e`), personal
details (`1c`), direct identity (`3a`), weather (`2a`), emotion (`5a`), general
knowledge (`4f`), human experience (`5b`), humor (`5c`), logic/math (`4d`),
strings (`4c`), and surroundings (`2c`). These classes seed the candidate bank;
their observed success is not treated as causal evidence.

The fixed order is `daily, opinion, personal, direct`. This is frozen from the
four most frequent classes in the released GPT-4o strategy classifications
(495, 424, 418, and 156 tags respectively), not selected from D0 outcomes.

## Splits, surfaces, and episode count

- 8 development scenario families and 8 held-out scenario families.
- 256 episodes per family: exactly 128 human and 128 AI, deterministically
  shuffled.
- 4 policies × 16 families × 256 episodes = 16,384 primary policy episodes.
- Four-question budget; questions cannot repeat; no early stopping.
- Development families use `development` question/answer renderings. Held-out
  families use disjoint `heldout` renderings.
- Policies consume only question IDs, response categories, posterior values and
  likelihoods. They never parse rendering text. The held-out-surface result thus
  tests pipeline separation, not language understanding.

The final Track A test split is unrelated and must not be read.

## Policies

| Arm | Frozen behavior |
|---|---|
| B0 random | seeded uniform choice among unused questions |
| B1 fixed | prevalence-based order `daily, opinion, personal, direct` |
| B2 BED-EIG | exact one-step mutual information; deterministic question-ID tie break |
| B2-UoT | one-sample-per-type simulated answer entropy; secondary only |

All policies receive the same prior, exact calculator, budget, question bank and
matched response schedule. Only selection differs.

## Outcomes and estimand

The primary episode outcome is final natural-log loss of the authoritative
posterior for the true type. For comparator `c`, define paired improvement:

```text
delta_c = final_log_loss(c) - final_log_loss(BED-EIG)
```

Positive values favor BED. The unit of replication is the scenario family, not
individual turns. Point estimates average episodes equally. The 95% interval is
a paired percentile bootstrap that resamples held-out scenario families and
retains all matched episodes/policies within each sampled family (10,000
replicates, seed 20260822).

Secondary outcomes are final Brier score, accuracy at 0.5, entropy/information
gain per question, repeated-question count, ledger validity, and per-family
effects. Mean, median, IQR, and family range are reported for paired log-loss
improvements.

## Pre-registered Gate 2A decision

Gate 2A is evaluated on the eight held-out families only. BED-EIG passes only if
**both** comparisons (versus B0 random and B1 fixed) satisfy all of:

1. mean paired final-log-loss improvement is at least `0.05` nats;
2. the family-clustered bootstrap 95% lower bound is above `0`;
3. at least 7 of 8 family-specific mean improvements are positive; and
4. every output ledger is valid, with zero repeated questions and exact Bayes
   updates agreeing to absolute tolerance `1e-12`.

Development-family improvements must also be positive in both comparisons as a
directional guardrail, but do not receive a threshold or interval gate.

- **PASS:** every condition holds. This permits designing the later D0 SFT stage;
  it does not run or authorize training automatically.
- **FAIL:** either held-out point estimate is non-positive or ledger integrity
  fails.
- **INCONCLUSIVE:** positive effects fail a size, interval, family-consistency,
  or development-direction condition. No D0 SFT proceeds.

UoT-style results cannot rescue a failed primary BED gate and are not included in
the gate multiplicity.

## Durability and inspection

The runner writes one trajectory JSONL row per policy episode with fsync, plus an
atomic state file containing config hash, seed, completed IDs, resume command and
failure state. A resume validates existing row IDs and config hashes before
skipping them. At least two trajectories per policy and the worst held-out BED
failures must be read before interpreting aggregate results.
