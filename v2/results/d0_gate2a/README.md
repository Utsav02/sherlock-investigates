# D0 Gate 2A result

**Run:** 2026-08-22

**Protocol:** `v2/D0_GATE2A_PROTOCOL.md` (frozen in `77ee76e`)

**Implementation:** `v2/scripts/d0_gate2a.py`

**Config:** `v2/configs/d0_gate2a_v1.json`

## Decision

**Formal Gate 2A: PASS.** Exact BED-EIG beat both frozen comparators on held-out
families under every preregistered condition.

| held-out comparison | paired log-loss improvement (nats) | family-clustered 95% interval | positive families |
|---|---:|---:|---:|
| random − BED | 0.224 | [0.200, 0.246] | 8/8 |
| fixed − BED | 0.238 | [0.159, 0.314] | 8/8 |

The development effects were also positive: 0.214 nats versus random and 0.204
versus fixed. BED held-out final log loss was 0.302 versus 0.526 random and 0.541
fixed; accuracy was 0.867 versus 0.703 and 0.695. All 16,384 expected policy
episodes were present exactly once, and full deterministic replay found zero
Bayes, repetition, provenance, or coverage failures.

UoT-style was descriptive only. Its held-out final log loss was 0.513; BED's
paired advantage was 0.211 nats [0.139, 0.285] across the family bootstrap.

## Critical interpretation

The formal pass is narrower than “active questioning works.” A mandatory
post-run trajectory audit found that BED used **exactly one question sequence
per family** across all 256 episodes. The symmetric likelihood construction
preserves the ranking of questions as the posterior changes. BED therefore did
not condition its question order on observed answers; it acted as a
family-specific oracle ranking.

That is useful evidence that the simulator, likelihood ledger, posterior update,
and exact question-value calculation are internally consistent. It is also a
planning failure in the construct: the benchmark was meant to exercise active
selection, but its primary policy reduces to non-adaptive prioritization. The
large advantage is unsurprising because BED receives each family's true
likelihood table while random and the global fixed order do not.

Accordingly, this PASS permits design work only. It does **not** justify D0 SFT
yet on scientific grounds, establish response-conditioned adaptation, test
language understanding, demonstrate family transfer, or support any claim about
real investigators. Before spending GPU time, a revised D0 should include
asymmetric or state-dependent likelihoods that force question rankings to branch
after different answers, and should freeze an explicit adaptation diagnostic.
This recommendation is stricter than the original gate; it is labelled post hoc
rather than being smuggled into the decision rule.

The worst BED trajectories also show the expected cost of confident evidence
under a noisy model: held-out log loss reached 4.951 when several individually
unlikely cues aligned against the true type. This is not a ledger error, but it
shows why calibration and tail-risk reporting remain necessary even in D0.

## Artifacts

- `trajectories.jsonl` — 16,384 complete ledgers; SHA-256
  `207d9ecf91e93c1a1b04e1f6b3903f46d80668f87afc1dc6fbed6615f58f4a20`
- `run_state.json` — completed durable-run state (its resume timestamp changes on
  a no-op resume; the trajectory artifact above remains byte-identical)
- `result.json` — preregistered inference plus post-hoc adaptation diagnostic
- `inspection.json` — one human and AI trajectory per policy plus the eight
  worst held-out BED losses

Reproduce with `make v2-d0-run` and `make v2-d0-analyze`. The runner rejects
duplicate, foreign, config-mismatched, or non-reproducible rows on resume.
