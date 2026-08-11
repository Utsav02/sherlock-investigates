# Dose-response curve — full 22-checkpoint run (2026-08-08)

**Status:** Result captured. **Weights lost, measurement kept.** The finding is
confirmed and strengthened relative to 2026-08-07.

Data: `results/analysis/dose_curve_20260808_204827.json` (reconstructed from the
notebook output — see "Process failure" below).

---

## What was run

One full-canon QLoRA run on a Kaggle T4, `configs/kaggle_t4_dosecurve.yaml`,
every checkpoint retained (`save_total_limit=null`), then think-block closure
scored at all 22 checkpoints + final with `scripts/eval/dose_curve.py` (seeded,
n=8 prompts/checkpoint, temp 0.7, `INITIATOR_SYSTEM_THINKING`).

- Base: `unsloth/DeepSeek-R1-Distill-Qwen-7B`, fp16 on T4
- 12,999 examples → 1,636 blocks → **103 steps, 1 epoch**, ~5h04m train
- `train_loss = 0.8482460952499538` — **bit-identical to the 2026-08-06 run**,
  so these checkpoints are deterministically reproducible.

## The curve

| step | closure | rate | Wilson 95% CI | mean tokens |
|---|---|---|---|---|
| base | 8/8 | 1.00 | [0.68, 1.00] | 516 |
| 5 | 8/8 | 1.00 | [0.68, 1.00] | 474 |
| 10 | 8/8 | 1.00 | [0.68, 1.00] | 423 |
| 15 | 7/8 | 0.88 | [0.53, 0.98] | 424 |
| 20 | 3/8 | 0.38 | [0.14, 0.69] | 321 |
| 25 | 4/8 | 0.50 | [0.22, 0.78] | 496 |
| 30 | 5/8 | 0.62 | [0.31, 0.86] | 478 |
| 35 | 4/8 | 0.50 | [0.22, 0.78] | 372 |
| 40 | 7/8 | 0.88 | [0.53, 0.98] | 431 |
| 45 | 3/8 | 0.38 | [0.14, 0.69] | 358 |
| 50 | 3/8 | 0.38 | [0.14, 0.69] | 296 |
| 55 | 3/8 | 0.38 | [0.14, 0.69] | 304 |
| 60 | 4/8 | 0.50 | [0.22, 0.78] | 307 |
| 65 | 1/8 | 0.12 | [0.02, 0.47] | 267 |
| 70 | 3/8 | 0.38 | [0.14, 0.69] | 427 |
| 75 | 5/8 | 0.62 | [0.31, 0.86] | 443 |
| 80 | 2/8 | 0.25 | [0.07, 0.59] | 318 |
| 85 | 4/8 | 0.50 | [0.22, 0.78] | 380 |
| 90 | 4/8 | 0.50 | [0.22, 0.78] | 391 |
| 95 | 6/8 | 0.75 | [0.41, 0.93] | 360 |
| 100 | 4/8 | 0.50 | [0.22, 0.78] | 344 |
| 103 | 2/8 | 0.25 | [0.07, 0.59] | 266 |
| final | 3/8 | 0.38 | — | 413 |

**Pooled:** early (≤35) **39/56 = 0.70 [0.57, 0.80]** vs late (≥45)
**44/104 = 0.42 [0.33, 0.52]**, Fisher exact two-sided **p = 0.0015**.

## What it tells us

1. **The finding replicates and is stronger.** Degradation with dose is
   significant (p=0.0015, essentially the 2026-08-07 p=0.0014). The onset is
   *earlier* than the coarser prior run implied: **step-20 is already 3/8**
   (~655K unique tokens — below the ~1M-token dose the threshold literature
   says you need to move a reasoning prior). There is no checkpoint that is
   both format-intact and dosed heavily enough to plausibly produce the effect.

2. **Same mechanism, confirmed a third time.** Failures are not truncation
   (`trunc` ≈ 0 throughout); mean token count *shrinks* on failing checkpoints
   (516 → ~270). The model emits a shorter reasoning blob and simply never
   closes `</think>` — catastrophic forgetting of the RL-trained format under
   raw-text causal LM, exactly as hypothesised on 2026-08-06.

3. **The statistics reframe was vindicated by this run.** Per-point numbers
   moved between the two bit-identical-loss runs (step-25: 6/8 → 4/8; step-35:
   6/8 → 4/8) — pure n=8 sampling noise — while the pooled conclusion held to
   three decimals. This is precisely why we do not headline a single "collapse
   step": the curve is a noisy decline (step-40 rebounds to 7/8, step-95 to
   6/8), not a clean cliff, and no single checkpoint can be certified safe at
   this n.

## Process failure — the 4th loss, and the fix

The HF token was never in the environment (the Kaggle Secret wasn't attached),
so despite persistence being "built in," **nothing was uploaded** — the run
warned twice and trained for 5 hours anyway, then the session was closed and
`/kaggle/working` was wiped. The 21 checkpoints and final adapter are gone
(reproducible, since the loss is bit-identical, but at the cost of another 5h).

Root cause: persistence was **fail-open** — a missing token was a warning, not a
stop. On a project whose entire history is missed warnings, that is the wrong
default. Fixed this session (see Decision Log 2026-08-08):

- `train_lora.py` now **aborts before training** if `hf_repo_id` is set and no
  token is reachable (escape hatch: `ALLOW_UNPERSISTED=1`).
- The driver notebook's preflight `assert`s the token before the 5h burn.
- `dose_curve.py` and the notebook's closing message are now **honest** — they
  say "NOT persisted, local only" when that is the truth, instead of "safe to
  close the session."

The measurement survived only because it was in the printed output. That is luck,
not a system; the fail-closed change removes the luck.

## Next

Re-run is now safe and cheap (free, ~5.5h) with the token set — the fail-closed
gate guarantees the checkpoints persist this time. But the *scientific* question
is already answered twice: **QLoRA continued-pretraining on raw prose degrades
R1-Distill's reasoning format at or below the dose needed for behavioural
transfer.** The open moves are (a) break the steps-vs-tokens confound and
(b) test base-model-generated rehearsal data — both free — before any 14B spend.
