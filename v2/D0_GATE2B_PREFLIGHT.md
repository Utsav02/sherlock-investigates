# D0 Gate 2B draft structural preflight

**Date:** 2026-08-26

**Status:** historical structural preflight for an unexecuted draft. The owner
stopped the active-investigation extension on 2026-08-29; Gate 2B performance
was not executed.

**Draft config SHA-256:**
`e3a69d05e6c1229b83f41bac5d555f845237c6837477dd8193fb55cd7c87f99d`

The hash changed from the pre-stop draft only because its non-operative
`design_status` field was changed to `historical_unexecuted_draft_programme_stopped`;
the generator and likelihood specification were not altered.

## Scope of this preflight

This preflight resolved the deterministic family generator and enumerated only
the optimal finite-horizon policy trees needed to test structural branching. It
did not compute the family-aware open-loop comparison, the global open-loop
comparison, any Gate 2B effect, bootstrap interval, or decision.

The one-off checker lived in `/tmp`, not the repository. It is not a Gate 2B
runner and is not a deliverable implementation.

## Checklist

| Invariant | Executed check | Result |
|---|---|---|
| Draft config parses | `jq`/Python JSON load | PASS |
| Family arithmetic | 16 indices = 8 development + 8 held-out; 12 unique questions × one role each | PASS |
| Deterministic generator | SHA-256 role ordering and strength selection, seed `20260826` | PASS |
| Strict normalized likelihoods | resolved all `16 × 12 × 2` categorical rows; tolerance `1e-12` | PASS; zero failures |
| Resolved probability range | inspect all 1,152 cells | PASS; `[0.05, 0.85]` overall |
| Exact policy-tree coverage | enumerate `1+3+9+27+81=121` reachable states/family | PASS; 1,936 total |
| Answer-dependent sibling actions | inspect all nonterminal sibling histories | PASS in 16/16 families |
| Eventual subset changes | compare terminal-subset collections below each sibling | PASS in 16/16 families |
| Practical branch threshold | sweep owner options | UNRESOLVED; results below |
| Gate 2B performance | prohibited before freeze | NOT RUN |
| Credentials/endpoints/GPU | none required | N/A |
| Launch/recovery | no run authorized | N/A |

## Design-iteration record

The first candidate used the same seed and family indices but a broader mix of
routers, specialists, and balanced channels. Its config hash was
`96330482abbd6b0907d8008f28db13b78e8efaeb2704efff0a18a6a3c2611f8b`.
It had subset-changing branches in 13/16 families, but no family jointly met an
absolute child-history probability of `0.05` and regret of `0.005` nats.

That candidate was rejected at structural preflight before any adaptive-versus-
open-loop outcome was calculated. The generation seed and all indices were
retained. The declared archetype system was revised to two routers plus five
one-sided probes in each direction. No individual family was replaced or edited.
This failed candidate is also recorded in the draft config.

## Current generated-family manifest

Counts below are threshold-free except for numerical uniqueness (`1e-10`).

| family | answer-dependent sibling pairs | subset-changing sibling pairs | manifest SHA-256 prefix |
|---|---:|---:|---|
| dev_asym_00 | 20 | 14 | `c95852bb7e4e` |
| dev_asym_01 | 21 | 17 | `19a29a54fbc5` |
| dev_asym_02 | 22 | 7 | `45576d524105` |
| dev_asym_03 | 14 | 12 | `8454f0352ac7` |
| dev_asym_04 | 20 | 14 | `da6a4ee1034d` |
| dev_asym_05 | 21 | 5 | `479760d5b30e` |
| dev_asym_06 | 21 | 19 | `d62c6d5561dc` |
| dev_asym_07 | 20 | 4 | `1cab2b003dee` |
| hold_asym_00 | 23 | 9 | `b99b5ae8f6ad` |
| hold_asym_01 | 19 | 8 | `f1a0da9262e7` |
| hold_asym_02 | 18 | 13 | `b3bdc0acd0a0` |
| hold_asym_03 | 16 | 10 | `45f02639734c` |
| hold_asym_04 | 26 | 25 | `e4b064637cf0` |
| hold_asym_05 | 22 | 10 | `47f0e3fc16a4` |
| hold_asym_06 | 23 | 23 | `8b7d5c4755e2` |
| hold_asym_07 | 18 | 10 | `3a64cd1287c5` |

This establishes technical response-conditioned branching and eventual subset
changes. It does not decide how much reach and regret are scientifically enough.

## Structural threshold sensitivity

Each cell reports the number of families with at least one qualifying branch.
All rows below additionally require a conditional child-response probability of
at least `0.10`, unique actions, and a changed eventual subset.

| absolute child-history probability | wrong-sibling regret | development | held-out |
|---:|---:|---:|---:|
| 0.005 | 0.001 nats | 8/8 | 8/8 |
| 0.005 | 0.005 nats | 6/8 | 6/8 |
| 0.005 | 0.010 nats | 3/8 | 4/8 |
| 0.010 | 0.001 nats | 7/8 | 8/8 |
| 0.010 | 0.005 nats | 5/8 | 6/8 |
| 0.025 | 0.005 nats | 3/8 | 5/8 |

The individually recommended draft settings (`0.01` absolute probability,
`0.10` conditional probability, `0.005` regret, and 6/8 families) do **not** pass:
development reaches only 5/8. The config remains deliberately unfrozen rather
than weakening a threshold to fit this output or regenerating families.

## Threshold choices requiring owner decision

None of these values is derived from a natural constant or from final Gate 2B
performance. The examples translate their units.

### Absolute child-history probability

- `0.005`: about one occurrence per 200 matched episodes; admits late but real
  policy-tree branches.
- `0.010`: about one per 100; middle option.
- `0.025`: about one per 40; favors common branches and rejects much of the
  current tree.

For a four-turn, three-response tree, third-turn child histories average roughly
`1/27 = 0.037` before skew. The earlier proposal of `0.10` as an **absolute**
history threshold is therefore too severe for many legitimate late branches.
The draft separates absolute history probability from conditional response
probability.

### Conditional child-response probability

- `0.05`: excludes only very rare answer categories.
- `0.10`: recommended middle option.
- `0.20`: requires each sibling answer to be common at its parent.

This threshold alone does not ensure the parent itself is reachable, hence the
separate absolute threshold.

### Wrong-sibling continuation regret

- `0.001` nats: approximately a 0.1% multiplicative difference in geometric
  probability assigned to the truth; primarily excludes ties and trivial noise.
- `0.005` nats: approximately 0.5%; moderate practical requirement.
- `0.010` nats: approximately 1%; stringent for a single local action.

For example, if two sibling actions produce continuation losses `0.300` and
`0.305`, forcing the wrong action has `0.005`-nat regret even if both actions
would give the same 0.5-threshold classification.

### Family coverage

- `6/8`: permits two structurally weak families per split.
- `7/8`: permits one; mirrors Gate 2A consistency but has no new inferential
  justification.
- `8/8`: treats every family as a construct-validation fixture.

The structural coverage threshold is selected separately from the held-out
performance-consistency threshold, whose options are also 6/8, 7/8, or 8/8
strictly positive exact adaptation effects. Coverage describes prevalence in
this finite generated set. It is not a confidence level. Because dominance
already prevents negative exact effects, zero versus positive is the informative
distinction at performance time.

### Mean adaptation-value floor

- `0.005`, `0.010`, or `0.020` nats correspond to roughly 0.5%, 1%, or 2%
  multiplicative improvement in geometric probability assigned to the true type.
- Recommended draft option: `0.010` nats.

This threshold must be selected before the family open-loop values are computed.

### Adaptation share

Options are `10%`, `25%`, or `40%` of the total exact improvement from global
open-loop to finite-horizon adaptive BED. If total improvement were `0.040` nats,
these require `0.004`, `0.010`, or `0.016` nats to come specifically from answer-
conditioned adaptation. Recommended draft option: 25%.

### Generator-seed sensitivity

1. One frozen seed only: cleanest single finite design, no robustness statement.
2. Primary seed plus `20260827` and `20260828` as descriptive-only sensitivity;
   they cannot rescue the gate. **Draft recommendation.**
3. Require the gate to pass all three seeds: strongest generator robustness, but
   triples families and makes the gate a different experiment.

No auxiliary seed has been generated or inspected.

## Suggested coherent structural bundles

These are choices, not recommendations derived from the preflight result:

| bundle | absolute | conditional | regret | coverage | current status |
|---|---:|---:|---:|---:|---|
| construct-detection | 0.005 | 0.10 | 0.001 | 8/8 | passes structural draft |
| reach-focused | 0.010 | 0.10 | 0.001 | 7/8 | passes structural draft |
| consequence-focused | 0.005 | 0.10 | 0.005 | 6/8 | passes structural draft |
| strict combined | 0.010 | 0.10 | 0.005 | 6/8 | fails structural draft (5/8 development) |

The `reach-focused` bundle accepts smaller local regret; the `consequence-focused`
bundle accepts rarer branches. Choosing between them is a scientific judgment.

## Exact implementation and runtime plan — not authorized yet

If later approved:

1. Resolve and hash all likelihood tables from the config.
2. Implement two independent posterior calculators (log-odds and normalized
   direct multiplication) for replay agreement.
3. Enumerate the 495 four-question subsets for each open-loop oracle.
4. Compute finite-horizon BED with memoized Bellman recursion and independently
   brute-force reduced fixtures.
5. Enumerate only policy-reachable trees for authoritative inspection.
6. Integrate exact risks and decompositions; then apply family-level summaries.
7. Run adversarial fixtures before any result command.
8. Optionally create a small stratified renderer/replay sample; never use it for
   gate estimates.

The structural preflight enumerated all 1,936 primary reachable states in about
6.5 seconds in an unoptimized one-off Python checker. Expected production CPU
runtime is roughly 30–120 seconds, conservatively under three minutes including
open-loop enumeration, all six policies, integrity checks, and bootstrap.

Expected authoritative storage:

- resolved config and manifests: below 0.5 MB;
- roughly 10,000 reachable deterministic-policy state records: about 3–10 MB as
  compact JSONL, less in a columnar format;
- aggregate results/inspection: below 1 MB;
- optional 200–400 sampled audit trajectories: about 0.5–2 MB.

Total expected size is approximately **5–15 MB**, materially below the earlier
65–100 MB sampled-trajectory estimate. Internal dynamic-programming cache states
need not be persisted as authoritative policy-tree nodes; their hashes and
independent verification results should be.

## Preflight decision

**CLOSED WITHOUT A GATE RESULT.** The proposed generator had real reachable,
answer-dependent, subset-changing branches in both splits, but the owner stopped
the active extension before threshold selection, freezing, implementation, or
performance evaluation. This is neither PASS, INCONCLUSIVE, nor FAIL. Any future
revival must begin with a new programme-level decision, not by selecting the
remaining thresholds in this draft.
