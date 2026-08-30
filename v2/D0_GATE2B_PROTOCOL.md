# D0 Gate 2B protocol — HISTORICAL UNEXECUTED DRAFT

**Status:** not frozen; the owner stopped the active-investigation extension on
2026-08-29. This protocol may not be implemented or run without a new explicit
programme-revival decision.

**Drafted:** 2026-08-26

**Scope:** exact synthetic mechanics only. No LLM inference, SFT, GPU, human
collection, estimated likelihoods, or Track A test access.

## 1. Purpose and claim boundary

Gate 2B asks whether a prospectively generated synthetic environment contains
reachable, decision-relevant answer-conditioned structure large and prevalent
enough to justify producing D0 policy targets. It does **not** ask whether an
optimal adaptive policy can beat a policy class nested inside it: that dominance
is mathematical. It does not establish language understanding, likelihood
estimation, unknown-family adaptation, real-active value, or transfer to real
respondents.

The historical Gate 2A PASS remains unchanged. Gate 2A validated exact Bayesian
replay and family-specific oracle prioritization under its frozen symmetric
model; it did not test response-conditioned adaptive questioning.

## 2. Exact generative model

The hidden identity is binary, `human` or `ai`, with prior 0.5. Each episode has
one known scenario family, twelve unused-once questions, three response
categories (`human_cue`, `neutral`, `ai_cue`), and a four-question budget.

For every family `f`, question `q`, type `t`, and category `r`, the environment
provides an exact, strictly positive categorical likelihood

```text
P(r | t, f, q).
```

Likelihoods may be asymmetric: neutral rates need not match across types and the
human/AI category rows need not be permutations of each other. Conditional on
`(type, family)`, the twelve potential question responses are independent. The
posterior after history `h` is therefore exact:

```text
P(t | h) proportional to P(t) * product P(r_i | t, f, q_i).
```

Question and response surfaces are unchanged from Gate 2A. Text renderings never
own the update; the resolved categorical table does. Development and held-out
rendering banks remain disjoint.

Latent traits, conditional question dependence, respondent state, and unknown-
family inference are deferred. This keeps a failed result attributable to the
minimal adaptive construct.

## 3. Deterministic family generation

The draft machine specification is `v2/configs/d0_gate2b_v1.json`. It records the
generation seed, explicit asymmetric channel archetypes, permitted strength
grid, family indices, SHA-256 assignment algorithm, and every retained family.

For each generator index 0–15:

1. Sort the twelve question IDs by SHA-256 of the frozen seed, index, and ID.
2. Assign the sorted IDs to a fixed role sequence: two broadly discriminative
   routers, five AI-sensitive one-sided probes, and five human-sensitive
   one-sided probes.
3. Derive a strength mixing coefficient from a second SHA-256 expression and the
   frozen grid.
4. Mix the selected archetype with the uniform categorical channel by the exact
   formula in the config.
5. Retain the family regardless of branching prevalence or adaptive value.

Indices 0–7 are development and 8–15 held-out. No family may be regenerated,
replaced, or manually edited after seeing policy values. Invalid probability
normalization, an incomplete assignment, or a manifest/hash mismatch blocks the
config. Insufficient branching fails structural preflight without replacement.
Low adaptation value is a scientific result and can never justify rejection or
regeneration.

The config records one earlier archetype draft that failed structural preflight.
The seed and family indices were retained; the archetype system was revised
before any open-loop/adaptive performance comparison.

## 4. Policies and dominance relations

| Arm | Information and optimization |
|---|---|
| adaptive finite-horizon BED | Knows the family and exact table; chooses a history-dependent policy minimizing expected terminal log loss over the remaining budget |
| family open-loop oracle | Knows the family; chooses the optimal four-question subset before any answers |
| global open-loop oracle | Knows the uniform mixture of families in the evaluated split but not the realized family; chooses one optimal four-question subset |
| one-step EIG | Knows the family; greedily maximizes immediate mutual information after each answer |
| fixed | Gate 2A order: `daily, opinion, personal, direct` |
| random | Uniform sampling without replacement; integrated exactly rather than estimated from episodes |

At history `h` with `b` questions remaining, finite-horizon BED uses

```text
V(h, 0) = entropy(P(AI | h))
V(h, b) = min over unused q of sum_r P(r | h, q) V(h + (q,r), b-1).
```

The open-loop class is a subset of the adaptive class. Consequently, for every
family,

```text
expected_loss(adaptive finite-horizon BED)
    <= expected_loss(family open-loop oracle).
```

Similarly, finite-horizon BED cannot be worse than one-step EIG; the family
open-loop oracle cannot be worse on average than the global open-loop oracle;
and the global open-loop optimum cannot be worse than fixed or expected random.
Any violation beyond `1e-12` is an implementation/exactness failure, not a
negative scientific result. Positive differences are partly structural. The
scientific quantities are their magnitude, family prevalence, reachability,
and the regret attached to actual branches.

## 5. Structural preflight — separate from Gate 2B performance

Performance evaluation is forbidden unless the frozen config passes structural
preflight. Preflight enumerates the exact finite-horizon policy tree and all
positive-probability outcomes under that policy.

A qualifying branch is a pair of sibling histories that:

- share the same action/history prefix and differ in the latest response;
- have the same unused question set;
- select different next questions;
- have unique optimal actions separated by more than the numerical tie tolerance;
- exceed the selected absolute-history and conditional-response probability
  minima;
- impose at least the selected continuation-value regret when the action from
  one sibling is forced at the other and optimal control resumes afterward; and
- yield different collections of possible final question subsets.

The preflight must additionally verify normalized strict likelihoods, exact
posterior replay, Bellman recursion against an independent exhaustive fixture,
deterministic policy trees, tree probability mass one, and adaptive dominance.

Insufficient qualifying-family coverage is `PREFLIGHT FAIL`. It is not `FAIL` or
`INCONCLUSIVE` on the Gate 2B scientific outcome, because the performance
evaluation never starts.

## 6. Authoritative artifacts

Exact integration, not sampled episodes, owns every Gate 2B estimate. Required
authoritative artifacts are:

- source config and fully resolved per-family likelihood tables with hashes;
- exact reachable policy trees for deterministic policies;
- enumerated reachable histories, posteriors, predictive probabilities, chosen
  actions, runner-up actions, action gaps, and continuation regrets;
- exact per-family expected final log loss for every policy;
- policy/config/implementation hashes and deterministic replay report.

Sampled trajectories have no gate authority. If approved, a small stratified
sample will be used only to check renderer provenance, durable resume, ledger
replay, and human inspection. Its count remains an owner decision.

## 7. Value decomposition and reporting

No combined headline improvement is permitted. Report separately:

```text
family knowledge
  = loss(global open-loop) - loss(family open-loop)

answer-conditioned adaptation
  = loss(family open-loop) - loss(finite-horizon BED)

lookahead
  = loss(one-step EIG) - loss(finite-horizon BED)

competent non-adaptive planning
  = loss(fixed or expected random) - loss(global open-loop).
```

For every decomposition report the exact effect for all sixteen families, plus
development and held-out mean, median, minimum, and range. For held-out families
also report a 10,000-replicate family bootstrap interval and the leave-one-family-
out mean range.

The bootstrap describes sensitivity to the finite generated family set. It is
not episode-level Monte Carlo uncertainty, uncertainty about a within-family
exact value, or evidence of generalization to real respondents.

Generator-seed sensitivity is unresolved. The options are one frozen seed; two
additional prespecified descriptive seeds that cannot rescue the gate; or a gate
that must pass all three seeds.

## 8. Draft decision rule — thresholds unresolved

All practical thresholds in the draft config are owner-selected judgments and
remain `selected: null`. Gate 2B cannot be frozen or run until they are selected.

After a successful structural preflight:

- **PASS:** exact integrity and dominance checks pass; held-out mean adaptation
  value meets the selected practical floor; the family-bootstrap lower endpoint
  is above zero; the selected held-out family-consistency rule is met; the
  development direction is positive; and adaptation meets the selected share of
  total global-open-loop-to-adaptive improvement.
- **INCONCLUSIVE:** integrity and direction are valid and mean adaptation value is
  positive, but a practical-size, finite-family sensitivity, family-consistency,
  development, or adaptation-share requirement is missed.
- **FAIL:** any exactness/integrity/dominance violation; or non-positive mean
  adaptation value after structural preflight. An exact negative value beyond
  tolerance diagnoses implementation failure. A zero value despite a verified
  positive-probability, positive-regret branch is also internally inconsistent.

Neither random, fixed, global open-loop, nor one-step EIG may rescue a failed
primary comparison.

## 9. Required adversarial tests before any run

Implementation approval must add fixtures for:

1. asymmetric tables whose channels remain ordered and never cross;
2. a technical branch with zero terminal utility advantage;
3. different orderings that produce an identical final subset;
4. unreachable and below-threshold histories;
5. floating-point near-ties on either side of the tie tolerance;
6. subtype or family information that leaves identity loss unchanged;
7. myopic EIG branching that exact finite-horizon BED rejects;
8. equality of adaptive and open-loop value in a non-adaptive control family;
9. adaptive value never below open-loop value beyond tolerance;
10. independent brute-force agreement with posterior, open-loop, and Bellman
    calculations on a reduced question bank.

## 10. Stop and authorization rules

This draft does not authorize implementation. After owner threshold selection,
the protocol and config must be marked frozen and hashed in an append-only
decision entry before writing the runner. A later implementation approval still
does not authorize D0 SFT, GPU use, human data, Track A test access, or a real-
active claim.
