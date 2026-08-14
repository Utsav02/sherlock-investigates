# Behavioural check: NO reasoning shift at step-50 (the real read)

**Date:** 2026-08-14. **Status:** Decisive negative — reading the think blocks,
base and the low-rank step-50 adapter reason the same way. The Phase-1
precondition (a *distinguishable reasoning prior*) is **not met** at step-50.

Evidence: `results/analysis/thinking_shift_20260814_171042_transcript.md`
(paired base/fine-tuned `<think>` blocks on the 30-prompt probe set, greedy),
structured JSON on HF (`utsvsngh/sherlock-dosecurve-results`).

---

## What the register table said (and why it misled)

| category | deduction Δ | hedging Δ |
|---|---|---|
| NEUTRAL (control) | ≈0 | −6.84 |
| DEDUCTION_INVITING | ≈0 | −8.18 |
| REASONING_REQUIRED | −0.83 | −0.88 |

At face value: no rise in deductive markers anywhere, and a hedging drop that is
about the *same* on the control as on deduction prompts — i.e. a generic style
nudge, not a deduction-specific shift.

**But the table is partly an artifact.** On prompts 16 and 18 the fine-tuned
model produced **no think block at all**, so its "hedging = 0.0" is *absence*,
not confidence — and those zeros pull the DEDUCTION_INVITING hedging delta down,
flattering the result. Across the transcript, ~18 of 60 think-block slots are
empty under greedy on these longer prompts; the fine-tuned adapter dropped the
block on 16/17/18 where base kept it on 16/18 (mild closure degradation on this
prompt distribution — note this is greedy on open-ended deduction prompts, not
the temp-0.7 conversational openers the 8/8 closure figure came from).

## Reading the actual think blocks

Where both models produced a block (prompts 10, 11, 12, 13), the reasoning is
**strikingly similar**: both walk cue-by-cue, hedge heavily ("probably", "could
indicate", "maybe"), and summarise. Neither reaches a confident identity
deduction. Prompt 10 (tanned hands, pale face, calluses, checking a watch):

- base ends *"I should probably not overthink it… I'll let him be."*
- fine-tuned ends *"I should probably ask him if he's okay… I can't be sure
  without asking."*

Neither says "a retired sailor." The fine-tuned trace, if anything, drifts
*away* from the detective register toward concern/helping. No deductive leap, no
Holmes voice.

## Verdict and why

**Behaviourally TOO_WEAK.** Perplexity said RESCUED, but that was ~34pp generic
prose-LM recovery + ~10pp Holmes-distributional; the think blocks confirm the
~10pp does **not** translate into a visible reasoning shift. All three signals
agree: perplexity decomposition, marker deltas, and the transcripts.

**Mechanistic reason (the useful finding):** the training corpus is the Holmes
*canon* — Watson *narrating* cases, dialogue, prose — **not** transcripts of
step-by-step deductive reasoning. Continued-pretraining on it teaches the model
to *predict detective prose* (hence the PPL drop) but gives almost no signal for
"when *you* reason privately, reason like Holmes." The model was trained on
*descriptions of* deduction and expected to shift its *own* deduction — a channel
mismatch. (Note: the augmentation's CHAIN/QA framings, which are closer to
reasoning shape, are flattened to raw text at training time — Decision Log
2026-08-06 — so even they were never trained as reasoning.)

## The fork

1. **Cheaply check step-103** (highest Holmes-specific excess, +9.7% vs step-50's
   +8.1%; closure still 7/8). ~15 min, no training. Low expected payoff — likely
   still null or marginal — but it closes the "maybe a higher dose shifts it"
   question for free.
2. **Rehearsal** — mix base-model-generated think blocks (actual reasoning
   traces, in the right channel) into training. This directly targets the channel
   mismatch above, and is the mitigation with a mechanism for producing a
   *behavioural* shift while keeping the format. More involved; contamination
   caution (blocks must be base-model-generated, not authored).
3. **Reframed writeup** — the arc is now a complete, publishable methods story:
   standard-rank QLoRA destroys the RL format (weight-movement, not breadth);
   low rank preserves it but at the format-safe dose the effect is mostly generic
   prose recovery with no reasoning shift, because raw narrative prose is the
   wrong channel for shifting private reasoning.

Recommended: run (1) for completeness (free), and draft (3) in parallel; treat
(2) as the one remaining rescue attempt if a behavioural shift is still wanted.
