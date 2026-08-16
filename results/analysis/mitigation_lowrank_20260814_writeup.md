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

## H2 pulled — the effect is MOSTLY GENERIC, not Holmes-specific

The run computed WikiText-2 perplexity (the H2 "did general language survive"
guardrail) but the driver did not print it. Pulled from the HF JSON after the
fact, it changes the reading of the effect substantially:

| step | Holmes drop | **WikiText drop** | Holmes-specific excess | Holmes/Wiki ratio |
|---|---|---|---|---|
| base | — | — | — | 1.122 |
| 10 | +26.3% | **+17.1%** | +9.1% | 0.998 |
| 20 | +37.7% | **+32.8%** | +5.0% | 1.039 |
| 50 | +42.5% | **+34.4%** | +8.1% | 0.984 |
| 103 | +43.8% | **+34.1%** | +9.7% | 0.956 |

**WikiText perplexity dropped ~34% too** — the model did NOT forget general
English; it got much *better* at predicting all prose. The mechanism is clear:
base R1-Distill is an RL-reasoning model, poor at raw next-token *prose*
prediction (it "wants" to emit reasoning), so continued-pretraining on any prose
restores strong prose LM and drops PPL on everything. The H2 gate (≤±5%) is
technically breached, but by improvement, not degradation.

**Consequence: ~34 of the 44 points of the Holmes PPL drop are this generic
prose-LM recovery; only ~10 points are Holmes-specific.** The clean
Holmes-specific signal is the *excess* — how much more Holmes dropped than
general text — and equivalently the Holmes/Wiki ratio falling from **1.122
(Holmes 12% harder) to 0.956 (Holmes ~4% easier)**, a ~15% relative
specialisation. Real, and — unlike the generic part, which saturates by step
~20 — it keeps growing slowly with dose (excess +5% → +9.7%). But it is a
fraction of the headline number.

So the "+44% effect" massively overstates Holmes learning. The mitigation's
RESCUED verdict used "PPL drop ≥ 5%" as the effect gate, and that gate fires
mostly on the generic recovery — it does **not** establish that the model
learned to reason like Holmes. Perplexity is now confirmed inadequate as the
effect measure, which makes the behavioural check **required**, not optional.

## Two caveats that still stand

1. **Perplexity is a proxy for a proxy.** Even the ~10pp Holmes-specific excess
   is distributional (predicts Holmes text better), not behavioural. The
   experiment's DV — Holmes-style deduction in the think block, and the
   commitment gap — needs the **behavioural** measure (probe separation, or
   inspecting think blocks on deduction-inviting prompts). This rescue clears
   the *format blocker*; it does not confirm the hypothesis.

2. **Don't use the final adapter.** Its closure is 3/8 (0.38) — much worse than
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

1. ~~Pull H2~~ — **done** (see above): WikiText dropped ~34%, so the effect is
   mostly generic prose-LM recovery; Holmes-specific signal is ~10pp.
2. Run the **behavioural** effect check on the step-50 checkpoint — now the
   decisive step, since perplexity is confirmed inadequate. Probe separation, or
   generate think blocks on deduction-inviting prompts and check whether the
   *reasoning* shifted (not just the prose). This is what upgrades "rescued
   blocker" to "hypothesis supported" — or refutes it.
3. Only if (2) shows a real reasoning shift, proceed to the conversation arm on
   the step-50 adapter.

**Revised odds:** the format rescue is solid; whether a ~10pp distributional
Holmes-specialisation produces a behaviourally-detectable reasoning shift is
genuinely uncertain — the behavioural check could still come back null, in which
case the finding is "low rank preserves the format but the achievable
Holmes-specific effect is too weak to shift reasoning," which points back toward
rehearsal (more targeted signal) or a reframed writeup.
