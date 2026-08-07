# QLoRA on raw prose destroys R1-distill's think-block closure

**Date:** 2026-08-06
**Status:** Controlled result, n=8 per arm. **Blocks the 14B spend.**

---

## The measurement

Three arms, one base model in VRAM, adapters swapped by name via
`PeftModel.set_adapter` / `disable_adapter`. Identical prompts (8 conversational
openers with `prompts.INITIATOR_SYSTEM_THINKING`), identical sampling
(temperature 0.7, `max_new_tokens=1200`), same session, same GPU.

| arm | steps | unique tokens | **closure** | truncated | mean tokens |
|---|---|---|---|---|---|
| base, no adapter | — | — | **8/8** | 0 | 451 |
| stage-0 | 30 | 311,252 | **8/8** | 0 | 528 |
| full canon | 103 | 3,352,033 | **1/8** | 1 | 417 |

"Closure" = the completion contains `</think>`, i.e. the model finished its
reasoning and moved to an answer. The chat template pre-opens `<think>`, so the
closing tag is the only detectable boundary.

## What it is not

Ruled out by the design, not by argument:

- **Not the model family, prompt, or token budget.** Base scores 8/8 on the
  identical prompts at the identical budget.
- **Not truncation.** 7 of the 8 fullcanon failures stopped *naturally* well
  under the 1200-token cap (mean 417). The model generates a normal quantity of
  reasoning and then simply never closes it.
- **Not the extractor.** Same `_resolve_think_block` scores all three arms, and
  it scores base and stage-0 at 8/8.
- **Not quantization or dtype.** One base load, shared across all three arms.

## What it is

Fine-tuning on the Holmes corpus at 103 optimizer steps destroyed a behaviour
that survived intact at 30 steps. The mechanism is almost certainly catastrophic
forgetting of the RL-trained reasoning format under raw-text causal LM: every
gradient step optimises "predict the next token of Victorian prose", and none
reinforce "emit `<think>`, close it, then answer".

**A collapse, not a gradient.** 8/8 → 8/8 → 1/8 across two dose points is
consistent with a threshold, but two points cannot distinguish a threshold from
a steep slope. The curve is unmapped.

## CONFOUND — do not over-read the dose axis

Stage-0 and full canon differ on **two** axes simultaneously:

| | corpus | epochs | steps |
|---|---|---|---|
| stage-0 | pilot (311K) | 3 | 30 |
| full canon | full canon (3.36M) | 1 | 103 |

So "103 steps broke it" and "3.36M unique tokens broke it" and "the full canon's
composition broke it" are all consistent with this data. **Optimizer steps is
the most likely culprit** — it is the quantity that directly controls how far
the weights move — but it is not established.

## Why this matters more than a bug

The experiment's premise requires a corpus large enough to shift a reasoning
prior. The threshold literature puts that at ~1M+ unique tokens (LIMA; Betley et
al.). **The dose that produces a behavioural effect appears to be the same dose
that destroys the channel the effect is measured through.** If that holds, the
design as specified cannot work, and no amount of GPU budget fixes it.

This is a genuine methodological finding about QLoRA continued-pretraining on
distilled reasoning models, and it is worth writing up whether or not the
original experiment survives it.

## What was lost, and the lesson

`save_total_limit=3` in `train_lora.py` kept only checkpoints 80/90/100, and the
Kaggle session wiped those anyway. **Had all ten checkpoints survived, this
run would already have given the full dose-response curve for free** — closure
rate at steps 10, 20, … 100. For a question about dose, checkpoints ARE the
experiment.

## Partial dose curve (2026-08-06) — run lost to the 12 h session cap

The dose-curve run trained successfully (bit-identical: `train_loss
0.8482460952499538`) and wrote 21 checkpoints, but the Kaggle session hit its
12-hour cap during evaluation and `/kaggle/working` was wiped. Three points were
read before it died:

| checkpoint | closure | mean tokens |
|---|---|---|
| base | 8/8 | 467 |
| step-5 | 8/8 | 472 |
| step-10 | 8/8 | 382 |
| step-103 (final, from HF) | **1/8** | 417 |

**The collapse is bracketed to (10, 103].** A healthy window demonstrably exists
at ≤10 steps, but 10 steps on this corpus is ~163K tokens seen — far below any
plausible effect dose, so this bracket does not yet answer the real question.

Process failure worth naming: training and evaluation were run as separate cells
hours apart, so the checkpoints had to survive a session boundary. They did not
need to. **Chain train -> evaluate in a single cell** — 4.6 h + ~40 min fits
comfortably inside one 12-hour session, and nothing durable has to cross a
boundary.

## Next, in order

1. **Map the curve.** Re-run the full canon with all checkpoints retained
   (`save_total_limit=None`), then measure closure at each. One 4.5 h run yields
   ~10 dose points and localises the collapse. This is the cheapest possible way
   to answer "where does it break", and it is free.
2. **Separate the confound.** A pilot-corpus run at 103 steps (3+ epochs) versus
   the full canon at 30 steps would separate "steps" from "unique tokens".
   Only worth doing if step 1 shows a sharp threshold.
3. **Then mitigations, in increasing cost:**
   - lower LR, or stop at the last pre-collapse checkpoint
   - lower LoRA rank / fewer target modules (less capacity to drift)
   - **rehearsal**: mix chat-formatted data carrying real think blocks into the
     corpus, so the format keeps being reinforced. Note the corpus ALREADY holds
     7,853 instruction-shaped examples (QA 2,208 / CHAIN 2,208 / WATSON 3,437)
     that `pack_into_blocks` currently flattens into raw text — the augmentation's
     stated purpose has never actually been exercised at the training step.
   - **Do NOT author synthetic think blocks.** Training the model to reproduce
     reasoning we wrote would contaminate the exact channel the experiment
     measures (Decision Log 2026-06-24). If rehearsal think blocks are needed,
     they must be generated by the base model itself.

**No 14B spend until step 1 is answered.** The gate did its job.
