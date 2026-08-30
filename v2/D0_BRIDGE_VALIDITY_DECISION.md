# D0 bridge-validity decision

**Status:** decided 2026-08-29 — stop the active-investigation extension. Gate
2B, counterfactual-fork work, D0 SFT, and downstream active-track evaluation are
not authorized.

**Date:** 2026-08-28

## Decision in one sentence

The owner chose to stop the active-investigation extension rather than add
another proxy between the evidence obtained and the programme's original
real-active target.

The Gate 2B drafts answer *how* to construct an adaptive finite-state benchmark.
They do not yet answer *why success on that benchmark would justify the next
programme step*.

## Objective trace

```text
Original target
  observable investigative behaviour in real human/AI conversations
    -> evidence seeking
    -> response-conditioned follow-up questions
    -> calibrated belief revision
    -> generalization

Cheap evidential decomposition
  Track A: real but passive
  Track B/D0: active but synthetic

Observed D0 result
  Gate 2A formally PASSed
    -> exact Bayesian ledger works
    -> family-specific oracle prioritization works
    -> responses never change the question sequence

Resolved fork
  Is a repaired synthetic mechanics test a useful bridge?
    -> yes, as theory/mechanics only: Gate 2B may be worth running
    -> only after conversational validity: test counterfactual follow-ups first
    -> NO SELECTED 2026-08-29: stop the active extension before SFT/GPU work
```

The programme has already established that a mathematically exact simulator can
reward question prioritization. It has not established that the simulator's
state, actions, response categories, or oracle likelihood access approximate
the problem an eventual language-model investigator must solve.

## What a positive Gate 2B result would and would not establish

A repaired Gate 2B could establish that, in a prospectively generated finite
Bayesian environment:

- answers lead to reachable posterior states with different uniquely optimal
  next questions;
- taking those branches has positive exact terminal log-loss value;
- that value remains after comparison with a family-aware optimal open-loop
  policy; and
- the effect is present across held-out generated families.

It would not establish that:

- the questions or answer distinctions are conversationally appropriate;
- a language model can infer the relevant response likelihoods from text;
- the synthetic branches resemble branches encountered with real respondents;
- the best synthetic action is a useful real follow-up;
- D0 SFT teaches a transferable policy rather than table imitation; or
- any benefit survives real transcript or real-active evaluation.

This is not a defect that a stronger Gate 2B threshold can repair. It is the
claim boundary of the proposed bridge.

## Options

| option | construct actually tested | strongest legitimate claim | principal weakness | next cost |
|---|---|---|---|---|
| **A. Run repaired exact Gate 2B** | Optimal response-conditioned control with known family and exact likelihoods | The generated environment contains consequential adaptive Bayesian structure | The environment is deliberately constructed to reward the policy and bypasses language understanding, likelihood learning, and unknown-family inference | Low: roughly 1–3 CPU minutes and 5–15 MB after implementation |
| **B. Counterfactual transcript-fork validity check first** | Whether different plausible answers to the same conversational prefix warrant different follow-up questions | The programme can operationalize response-sensitive questioning at the language/action surface | Without executable respondents it measures action appropriateness, not downstream causal question value; author judgments require provenance and agreement checks | Low–medium: a small authored/adjudicated development set; no GPU required |
| **C. Fixed real-transcript replay** | Belief updating as evidence accumulates in real text | Synthetic training may or may not transfer to passive real evidence | The investigator cannot choose a question, so this cannot validate an active policy | Already specified as Gate 3B, but only meaningful after a training intervention exists |
| **D. Stop the active extension** | No new active construct | The methodological failure study remains the honest deliverable | The original real-active question remains unanswered | None; avoids unjustified SFT and benchmark work |

### A. Repaired exact Gate 2B

This option is internally clean. The asymmetric, conditionally independent
model is the smallest exact construction that can produce policy crossover, and
the family-aware open-loop oracle correctly isolates adaptation from family
knowledge. Exact integration, adversarial degeneracy tests, and deterministic
family generation make the result reproducible.

The scientific limitation is not implementation quality. Adaptive BED weakly
dominates open-loop control by construction, and the generator intentionally
contains question channels capable of crossing as the posterior changes. A
positive magnitude shows how much the chosen synthetic distribution rewards
adaptation. It does not independently validate that distribution as a model of
investigation.

Gate 2B is therefore justified only if **exact adaptive Bayesian mechanics is an
independently valuable programme output**. It is not, by itself, sufficient
reason to create D0 SFT targets.

### B. Counterfactual transcript-fork validity check

For a shared conversational prefix, construct two or more plausible respondent
answers and require a different preferred next question for at least one pair.
The record should include the answer feature that changes the investigative
hypothesis, the candidate actions, the preferred action for each branch, and why
using the sibling action incurs a concrete opportunity cost.

This is closer to the target because it tests whether response-conditioned
branching survives translation into language and conversational actions. It can
also expose cases where a probability-table branch is mathematically valid but
pragmatically absurd.

It cannot supply exact causal information gain without executable respondents
or estimated likelihoods. It should therefore be treated as a **construct-
validity prerequisite**, not as a replacement performance gate. It must use only
development material or newly authored synthetic prefixes; the frozen Track A
test split remains untouched. Any human or model authorship must be recorded,
and proposed labels need independent adjudication before they become a gate.

### C. Fixed real-transcript replay

This remains useful for calibrated belief revision but is downstream and
passive. It cannot answer whether the model would have selected a better
question. Advancing it now would substitute an easier observable construct for
the original active one.

### D. Stop

Stopping is scientifically valid. The project already supports a methodological
failure account: prose training did not establish investigative behaviour,
Track A showed severe prompt dependence for in-corpus estimators, and Gate 2A
showed that a formally passing gate can miss its motivating construct.

Stopping should be preferred if neither exact Bayesian mechanics nor a small
counterfactual action benchmark is independently useful enough to publish,
reuse, or inform a later real-active design.

## Recommendation considered

If the owner retains the **real-active investigation** objective, choose option
**B before A**:

1. Design a very small, development-only counterfactual fork validity check.
2. Before authoring it, freeze what counts as a decision-relevant change in the
   best follow-up and how disagreements will be handled.
3. Use it to test whether the D0 question and response abstractions can express
   recognizable conversational branches.
4. Return to Gate 2B only if that check succeeds and exact mechanics remains a
   useful separate rung.
5. Even then, require a new owner decision between a Gate 2B PASS and any D0 SFT
   work; a mechanics PASS must not authorize training automatically.

If the primary deliverable is now the **methodological failure study**, choose
option **D**. The Gate 2B drafts should remain clearly marked as unexecuted design
work rather than being completed because they already exist.

Choose option **A directly** only if the narrowed output—an exact, synthetic
adaptive-control benchmark—is itself worth producing independently of language
or real-transcript transfer.

## Decision outcome

On 2026-08-29 the owner selected:

> **Stop active extension:** retain the methodological failure study and do no
> more D0 work.

This closes the hybrid programme's active extension. It is not a Gate 2B FAIL,
because Gate 2B was never frozen or run, and it does not alter the historical
Gate 2A PASS. It is a programme-level stop based on insufficient bridge validity.

Any future revival requires a new explicit owner decision beginning from the
original real-active objective. It must not resume automatically from the
existing Gate 2B drafts or treat their sunk effort as justification.

## Status of existing Gate 2B drafts

The following remain drafts and are not invalidated:

- `v2/D0_GATE2B_PROTOCOL.md`
- `v2/configs/d0_gate2b_v1.json`
- `v2/D0_GATE2B_ANALYTICAL_NOTE.md`
- `v2/D0_GATE2B_PREFLIGHT.md`

They are preserved as historical, unexecuted feasibility work. Their existence
creates no presumption that the stopped branch should be revived. No runner,
performance result, training authorization, or frozen protocol exists.
