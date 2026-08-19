# Stage A step 5 — source-coverage table

**Date:** 2026-08-18 · Discharges `v2/experiment_design.md` §18 Stage A step 5:
*"Produce a source-coverage table showing which v2 requirements remain missing."*

Written late: Stage B ran before this existed. That ordering error is recorded in
`STATUS.md` and is the reason this file leads with what is **missing**, not with
what is available.

## Verification key

| mark | meaning |
|---|---|
| **M** | **measured this project** from the downloaded files — cite in `data_inspection.md` |
| **D** | asserted by the source's own paper/card and quoted in our registry — not independently checked |
| **U** | **unverified** — from the design doc's candidate list or background knowledge; the source has not been downloaded or read this project |

Only the Jones & Bergen release carries **M** marks. Everything else is **D** or
**U**, which is itself the finding: one corpus was audited in depth and the rest
were never opened.

---

## The v2 requirements

From §1 ("no known drop-in dataset containing genuine humans, diverse adversarial
AI respondents, adaptive investigators, turn-level calibrated beliefs, and
verified SFT actions") plus the Track A/B needs in §5.

| id | requirement |
|---|---|
| R1 | genuine human respondents (not an AI prompted to seem human) |
| R2 | diverse adversarial AI respondents (≥3 substantive families) |
| R3 | adaptive investigator questions chosen by a real interrogator |
| R4 | **executable** respondents — can answer counterfactual "what if a different question" |
| R5 | turn-level beliefs / probabilities from the investigator |
| R6 | verified SFT actions (a target known to be correct, not merely fluent) |
| R7 | ground-truth identity labels |
| R8 | licence/consent clear enough for our intended use |
| R9 | model-family breadth sufficient for held-out-family claims |

---

## Coverage

| source | R1 | R2 | R3 | R4 | R5 | R6 | R7 | R8 | R9 | status |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|---|
| **JB 2025 three-party, 5-min** (primary) | ✅**M** | ⚠️**M** | ✅**M** | ❌**M** | ❌**M** | ❌**M** | ✅**M** | ⚠️**M** | ❌**M** | **audited, used, Track A concluded** |
| JB 2025 three-party, 15-min | ✅**M** | ⚠️**M** | ✅**M** | ❌**M** | ❌**M** | ❌**M** | ✅**M** | ❌**M** | ❌**M** | **Gate 0 UNRESOLVED — never read** |
| JB 2024 two-party | ?**U** | ?**U** | ?**U** | ❌**U** | ?**U** | ❌**U** | ?**U** | ?**U** | ❌**D** | not downloaded |
| Inverse Turing Bench | —**D** | —**D** | ❌**D** | ❌**D** | ❌**D** | ❌**D** | ✅**D** | ⚠️**D** | ❌**D** | derived from JB 2025; benchmark CSV seen once for the length determination, never registered |
| Human or Not? | ?**U** | ?**U** | ?**U** | ❌**U** | ❌**U** | ❌**U** | ?**U** | ?**U** | ?**U** | not downloaded; raw availability unclear |
| SPADE | ❌**U** | ?**U** | ❌**U** | ❌**U** | ❌**U** | ❌**U** | ✅**U** | ?**U** | ?**U** | not downloaded; largely synthetic customer-service dialogue |
| HC3 | ✅**U** | ❌**U** | ❌**U** | ❌**U** | ❌**U** | ❌**U** | ✅**U** | ?**U** | ❌**U** | not downloaded; static QA, single-model |
| HANSEN | ✅**U** | ?**U** | ❌**U** | ❌**U** | ❌**U** | ❌**U** | ✅**U** | ?**U** | ?**U** | not downloaded; spoken-text domains |
| **D0 (synthetic, to be built)** | ❌ | ❌ | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ | ❌ | **not built — Stage C** |

✅ present · ⚠️ present but limited · ❌ absent · ? unknown · — not applicable

---

## What this table says

**1. No existing source supplies R4, R5 or R6.** Not one. Every real corpus is a
fixed transcript: it cannot be asked a counterfactual question, carries no
turn-level belief, and offers no verified action to imitate. **D0 is the only
candidate for those three requirements, and it does not exist yet.** This is the
structural finding that justifies v2.1's split into a real-passive track and a
synthetic-active track, and it should have been on the record before Stage B.

**2. R2 and R9 are the primary corpus's binding weakness, and it is now measured.**
JB 2025 has two substantive families (OpenAI 581 games, Meta-Llama 412) plus ELIZA
(147) as a weak control. Rung 4 turned that from a caveat into a result: held-out
persona transfer fails, and the one system with no stylistic sibling (ELIZA) scores
below chance. Any future source must be assessed on family *and prompt* breadth,
not family count alone — a lesson this table did not previously encode.

**3. R8 is unresolved almost everywhere.** The primary corpus is Gate 0
CONDITIONAL (evaluation and local development only; no training, no
republication). The 15-minute study is unresolved and therefore unread, which is
exactly why no out-of-source holdout was ever available to Gate 1. Every other
candidate is at "verify terms" and none has a registry record.

**4. The audit stopped after one source.** Eight candidates in §8.1; one
downloaded, hashed, inspected and registered. That is the milder replay of V1's
failure mode: promising corpus found, work accelerated into analysis. The table
now makes the gap visible instead of implicit.

---

## Consequences for what happens next

- **Stage C (D0) is the right next move on the evidence**, not merely by plan
  order: D0 is the only route to R4/R5/R6, and Track A has now demonstrated that
  the best available real corpus cannot carry the generalization claim.
- **Do not open a new real corpus casually.** Each needs a registry record and a
  Gate 0 decision first (§16 Gate 0). The cost of the JB audit is the realistic
  estimate for any of them.
- **If a real-passive replication is ever wanted**, JB 2024 two-party is the
  cheapest untried candidate — same authors, same consent regime, probably the
  same licence question — but it is likely weaker on R9 and would not resolve the
  persona-transfer failure.
- **The 15-minute study remains the single highest-value unlock** (GPT-5 witness,
  median 22 messages vs 8, disjoint participants) and is blocked only on a Gate 0
  answer, which needs an author response rather than compute.
