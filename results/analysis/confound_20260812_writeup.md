# Confound resolved: STEPS (weight movement), not unique-token breadth

**Date:** 2026-08-12. **Status:** Clean result, and it refutes the breadth
hypothesis. Persistence worked end-to-end (every checkpoint + results on HF).

Data:
- pilot curve: `results/analysis/dose_curve_20260812_104622.json`
- overlay:     `results/analysis/confound_pilot103_vs_fullcanon.json`

---

## The design (one run)

Trained the **pilot corpus (311K unique) to ~110 steps by re-reading it 11×** —
identical model / LoRA / optimizer / cosine schedule to the full-canon dose
curve, so the ONLY difference between the two closure curves is corpus breadth
(1/11th). That fills the missing cell of a 2×2 whose other three cells already
existed (stage-0, full-canon early, full-canon late).

## The 2×2

| | early (≤35 steps) | late (≥45 steps) |
|---|---|---|
| **pilot (311K unique)** | 0.70 [0.57, 0.80] | **0.21 [0.15, 0.30]** |
| **canon (3.36M unique)** | 0.70 [0.57, 0.80] | 0.42 [0.33, 0.52] |

Four contrasts (Fisher exact):

| contrast | comparison | p |
|---|---|---|
| STEPS @ low breadth (pilot early vs late) | 0.70 → 0.21 | **2.7e-09** |
| STEPS @ high breadth (canon early vs late) | 0.70 → 0.42 | 0.0015 |
| BREADTH @ low steps (pilot vs canon, early) | 0.70 vs 0.70 | **1.0** |
| BREADTH @ high steps (pilot vs canon, late) | 0.21 vs 0.42 | 0.0012 |

## What it says

**It is optimizer steps / cumulative weight movement, not unique-token
breadth.** Two independent reads point the same way:

1. **Breadth does nothing at matched low dose.** Early, pilot and canon are
   *identical* (0.70 vs 0.70, p=1.0) despite an 11× difference in unique tokens.
2. **Steps break the format even at 1/11th the breadth.** The pilot collapses
   hard with steps (0.70 → 0.21, p=2.7e-09) — as hard as or harder than canon.

And the sharp, unexpected part: **at matched high steps the low-breadth pilot is
*worse* than the broad canon** (0.21 vs 0.42, p=0.0012). Re-reading a narrow
311K corpus 11× is *more* destructive to the reasoning format than seeing 3.36M
diverse tokens once. That is the opposite of "breadth hurts." The mechanism is
cumulative movement toward a narrow target: repeated passes over the same text
give a concentrated, consistent gradient that drives the weights further into
"complete Victorian prose" and further from "emit and close `<think>`". Diverse
tokens dilute that pressure.

## Decision consequence

The verdict licenses the **cheap** mitigations before the involved one:

- **Lower LoRA rank / fewer target modules** — the most mechanistically
  promising: constrain the adapter to a small subspace so the base weights (where
  the RL-trained format lives) are protected while a smaller adapter still picks
  up style. This is the single run to try next.
- **Lower LR / early stop** — worth trying but weaker; low-LR mostly rescales the
  same damaging trajectory rather than escaping it, and early-stop under-doses
  (closure is already sliding by step ~20, ~650K tokens, below the ~1M effect
  threshold).

**Tempered expectation, stated honestly:** this identifies the *lever*, it does
not prove a usable window exists. The core tension is unchanged — format damage
tracks weight movement, and enough movement to shift a reasoning prior may
inherently damage the format. Low-rank is a principled bet that it can be
decoupled; it is not a guarantee. **Rehearsal** (base-model-generated think
blocks mixed in) remains the robust fallback that works regardless.

## Standalone value

Independent of whether this experiment is rescued, the result is a clean methods
finding: *under QLoRA continued-pretraining on a distilled reasoning model,
catastrophic forgetting of the RL-trained output format is driven by optimizer
steps / weight movement, not by the breadth of the training distribution; and
repeatedly re-reading a small corpus is more destructive than a single pass over
a large diverse one at matched steps.*

## Next

The last **diagnostic** is done. The fork is now:
1. **One low-rank mitigation run** (e.g. rank 8, fewer modules) — cheap, free,
   now crash-safe; measures whether closure survives at an effect-relevant dose.
2. **Rehearsal** — more involved, robust.
3. **Negative-results writeup** — the finding is solid and worth writing up
   regardless of 1–2.

Recommended: run (1) AND start (3) in parallel — the writeup is valuable either
way, and (1) is one cheap run that decides whether the experiment is salvageable
without rehearsal.
