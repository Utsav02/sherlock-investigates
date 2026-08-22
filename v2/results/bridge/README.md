# Out-of-corpus detector bridge — result

**Run date:** 2026-08-22

**Protocol:** `v2/BRIDGE_PROTOCOL.md`, frozen before scoring

**Detector:** OpenAI RoBERTa-base GPT-2 detector, revision
`6cba99c003b711c7fe94f8a3aa2be35a792cb6fa`

**Data:** 850 non-empty paired train+dev games / 1,700 dialogues; frozen test
untouched

**Pre-registered verdict:** **PASS**, with the narrow interpretation below

## Result

The detector does **not** work zero-shot in its advertised direction on these
2025 chat transcripts. Raw overall dialogue AUROC is 0.463 and game accuracy is
0.454. On the persona cuts, raw game accuracy is 0.396 on Quinn and 0.409 on
minimal prompts. Its probabilities are also unusable as probabilities without
adaptation (overall dialogue Brier 0.416).

Every outer-fold Platt slope is negative. The target-corpus calibrator therefore
learns, from one prompt family only, to reverse the old detector's score ordering.
That reversed relationship transfers to the other prompt family:

| calibrator train → evaluation | game accuracy | dyadic participant 95% CI | component 95% CI | dialogue Brier |
|---|---:|---:|---:|---:|
| minimal → Quinn | 0.604 | [0.563, 0.645] | [0.548, 0.641] | 0.244 |
| Quinn → minimal | 0.591 | [0.531, 0.652] | [0.544, 0.650] | 0.247 |

Both pre-registered directions clear chance under both corrected intervals and
have Brier below 0.25, so the written gate returns PASS.

## What the pass means—and does not mean

It means an immutable, temporally prior external score contains weak but
repeatable information whose **inverse** relation to the label transfers across
the target prompt families after properly nested target calibration. This is
enough to justify proceeding to the cheap, no-training D0 Gate 2A mechanics.

It does not mean the 2019 detector recognizes modern AI text zero-shot, and it
does not validate individual accusations. The all-system calibrated result is
only 0.546 game accuracy and its component interval crosses chance
([0.488, 0.584]). Both transfer Brier scores sit close to the 0.25 no-information
reference. Only eight connected components remain after the split and empty-text
policy, so the component evidence is especially coarse. This is a deliberately
small bridge, not a new state-of-the-art detector result.

The old explanation “the corpus contains no transferable signal” is therefore
too strong. A better explanation is that the earlier in-corpus heads overfit the
small prompt/system support, while an external score carries a different but
weak signal. The bridge does not resolve whether that signal is semantic,
stylistic, model-era mismatch, or another corpus correlate.

## Consequence for the hybrid programme

The methodological failure study remains a standalone deliverable, including
the planning failures already recorded. The active extension may now begin
Stage C at Gate 2A only. No D0 SFT is allowed until a no-training active policy
beats its frozen random and fixed baselines. Any later claim of relevance to real
investigation additionally requires the mandatory real-replay Gate 3B in the
frozen protocol; synthetic success alone is insufficient.

## Reproduction

```bash
scripts/setup_bridge_env.sh
scripts/doctor_bridge_env.sh
.bridge-venv/bin/python -u v2/scripts/track_a_bridge.py score --resume
venv/bin/python v2/scripts/track_a_bridge.py analyze --bootstrap 1000
```

Durable artifacts:

- `openai_gpt2_detector_raw.jsonl`: text-free per-dialogue raw scores and input
  hashes; 1,700 unique rows, SHA-256
  `2de37dddcbb912d6b41e8031d9487faa69335b70ef9a9cdd98d97f72a8e7297c`.
- `openai_gpt2_detector_run_state.json`: environment versions, completion state,
  device and output hash.
- `openai_gpt2_detector_bridge.json`: nested folds, calibrator slopes, point
  estimates, corrected intervals and gate evaluation.

The full detector pass ran on CPU in 182 seconds after PyTorch advertised MPS
but rejected model transfer under macOS 26.3. The device fallback changes runtime,
not model weights or scoring arithmetic.
