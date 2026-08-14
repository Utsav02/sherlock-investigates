# Low-rank (r8) RESCUES the format — with one honest proxy caveat

**Date:** 2026-08-14. **Status:** Positive result — the first of the project.
Rank 8 preserves the `<think>` format at a dose that produces a large held-out
Holmes effect. **Rehearsal is not needed.** Persistence worked end to end.

Data (reconstructed into git from the Kaggle output; also on HF):
- closure: `results/analysis/dose_curve_20260814_042400.json`
- effect:  `results/analysis/effect_curve_20260814_065854.json`
- overlay: `results/analysis/mitigation_lowrank_r8.json`

---

## The one variable

Identical to the full-canon dose curve except **rank 32 → 8, alpha 64 → 16**
(scaling held at 2.0). So the closure curve overlays directly on the r32 curve
and the only difference is the dimension of the adapter subspace.

## Closure — dramatically better preserved at r8

| | early (≤35) | late (≥45) |
|---|---|---|
| **rank 8** | **0.96 [0.88, 0.99]** | **0.73 [0.64, 0.81]** |
| rank 32 (full canon) | 0.70 [0.57, 0.80] | 0.42 [0.33, 0.52] |

Fisher early-vs-late at r8: p = 0.0002. There is still some decay with steps,
but it is far gentler — exactly what the confound verdict predicted (fewer
directions to move the weights → less drift → format better protected).

## Effect — held-out Speckled Band perplexity

Base 32.54 → drops fast and saturates:

| step | 5 | 10 | 15 | 20 | 30 | 50 | 103 |
|---|---|---|---|---|---|---|---|
| PPL | 30.85 | 23.99 | 21.19 | 20.26 | 19.46 | 18.72 | 18.29 |
| drop | +5.2% | +26.3% | +34.9% | +37.7% | +40.2% | +42.5% | +43.8% |

Blows past the pre-registered H1 gate (≥5%): **+43.8%** at plateau.

## Verdict: RESCUED, and the window is comfortable

A checkpoint rescues iff closure ≥ 0.75 AND PPL drop ≥ 5% at once. Many do. The
**sweet spot is step ~50**: closure **8/8 (1.00)** AND drop **+42.6%** — full
format retention with essentially the plateau effect. You do not need to push to
high steps; steps 20–55 all pair closure ≥ 0.88 with effect ≥ +38%.

## Three caveats that must travel with this result

1. **Perplexity is a PROXY, and it saturates early.** The PPL drop is ~+35% by
   step 15 (~500–650K tokens) — *below* the ~1M-unique-token threshold the
   literature ties to reasoning-prior shifts. So most of the win is fast surface
   adaptation to Doyle's style (vocabulary, cadence, Watson narration), not
   proof that the model *reasons* like Holmes. "Learned Holmes prose" ≠ "reasons
   like Holmes in the think block." The experiment's actual DV — deduction
   behaviour / the commitment gap — still needs the **behavioural** measure
   (probe separation, or inspecting think blocks on deduction-inviting prompts).
   This rescue clears the *blocker*; it does not yet confirm the hypothesis.

2. **H2 guardrail unverified here.** The run computed WikiText-2 PPL (the
   "did general language survive" gate, ≤±5% drift) but the driver did not print
   it. Pull it from the effect-curve JSON on HF before trusting the effect as
   Holmes-specific rather than partly general degradation.

3. **Don't use the final adapter.** Its closure is 3/8 (0.38) — much worse than
   step-103's 7/8. Operate at a mid-window checkpoint (~step 50), not `final`.

## What changes

- **Rehearsal is off the table** for now — the cheap lever worked, no
  base-model-think-block generation pipeline needed.
- **The conversation experiment is unblocked.** For the first time there is a
  fine-tuned adapter that keeps the reasoning format AND has demonstrably
  absorbed Holmes.
- **The negative-results writeup is reframed**, not shelved: the finding is now
  "standard-rank (32) QLoRA continued-pretraining destroys a distilled model's
  RL-trained output format, driven by weight movement not corpus breadth, and a
  low-rank subspace (r8) rescues it" — a sharper, more useful methods result than
  a flat negative.

## Next, in order

1. Pull H2 (WikiText drift) from the HF effect JSON — confirm ≤±5%.
2. Run the **behavioural** effect check on the step-50 checkpoint — the real
   reasoning-shift signal (probe separation / think-block inspection on
   deduction prompts). This is what upgrades "rescued blocker" to "hypothesis
   supported."
3. If that holds, proceed to the conversation arm on the step-50 adapter.
