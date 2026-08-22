# Track A → D0 bridge protocol (frozen 2026-08-22)

This protocol was written before running the external detector. Stage C remains
paused until the bridge result is recorded. The Track A final test split remains
untouched.

## Question and smallest executable baseline

The bridge asks one narrow question: does a detector trained outside this corpus
retain any useful identity signal across the target corpus's minimal/persona
prompt shift?

The baseline is the 2019 OpenAI RoBERTa-base GPT-2 output detector at immutable
revision `6cba99c003b711c7fe94f8a3aa2be35a792cb6fa`. Its provenance and severe
domain limitation are recorded in
`v2/data/sources/registry/openai_gpt2_detector_2019.md`.

Only `train` and `dev` games are scored. Each retained game contributes its human
and AI witness text. Empty-witness games, including sides erased entirely by
redaction, are excluded. The raw score is the
model's `Fake` probability; neither labels nor target text are used to alter the
detector.

## Nested calibration

Calibration is Platt scaling on the logit of the frozen raw score. It is nested
inside the evaluation design:

1. Hold out one participant-connected component for evaluation.
2. On the remaining components only, select ridge strength from
   `{0.01, 0.1, 1, 10, 100}` by inner leave-one-component-out mean dialogue
   Brier score.
3. Refit the calibrator on all outer-training dialogues at the selected strength.
4. Predict the untouched outer component.

For persona transfer, the same nesting is used while training only on one prompt
family and evaluating only on the other. No evaluated component contributes to
calibrator selection or fitting. Model selection ties choose the larger ridge
penalty. Raw zero-shot scores are reported beside nested-calibrated scores so a
calibrator cannot manufacture a discrimination claim.

Additive outcomes use the corrected dyadic participant-cluster sandwich interval.
AUROC uses connected-component bootstrap. Component intervals are also reported
as sensitivity analyses.

## Pre-registered detector decision

The external detector is a usable bridge only if **both** minimal→persona and
persona→minimal nested evaluations have a game-accuracy 95% lower bound above
0.50 under both the dyadic-participant interval and the connected-component
bootstrap. Nested-calibrated dialogue Brier must also be below 0.25 in both
directions.

- Pass: all conditions hold. The detector may be used as one aggregate external
  probability authority in real replay, with its limitations attached.
- Fail: a point estimate is at or below chance or a Brier score is at/above 0.25.
- Inconclusive: point estimates clear the thresholds but either interval crosses
  them. This does not authorize a real-active claim; further external data or a
  separately pre-registered detector would be needed.

This gate concerns this baseline, not the universal possibility of detection.

## Mandatory D0 real-replay criterion

The programme is hybrid: the methodological failure study stands regardless of
later outcomes. If the project also claims relevance to real active
investigation, Gate 3B is mandatory rather than optional.

After Gate 3 passes on D0, replay the same fixed real `train+dev` transcripts,
turn by turn, through the frozen Track B base and the D0-trained investigator.
Neither arm may choose counterfactual questions. At every witness turn, record
the arm's probability that the witness is AI. Invalid/missing probabilities are
scored as `0.5` and counted; histories, decoding settings, seeds, checkpoints and
parsing rules are identical across arms.

The primary per-game outcome is the difference in area-under-the-log-loss curve:

`delta = mean_turn_log_loss(base) - mean_turn_log_loss(D0)`.

Positive values favour D0. Gate 3B passes only when the 95% lower bound for mean
`delta` is above zero under **both** (a) the dyadic participant-cluster sandwich
and (b) the connected-component bootstrap. A non-positive point estimate fails;
a positive estimate with either interval crossing zero is inconclusive. Failure
or inconclusive evidence permits reporting synthetic D0 mechanics, but forbids a
claim that D0 improves real investigation and blocks a real-active pilot.

Absolute scores are not compared with Track A detector arms. The frozen Track A
test split remains reserved for Gate 5 confirmation after all gates and analysis
code are frozen.

## Preflight and abort rules

- Expected scoring volume: 850 retained non-empty `train+dev` games, two
  dialogues each, for 1,700 detector calls. Sixty-one of 911 train+dev games are
  excluded because at least one released witness side has no usable text.
- Expected download: one 500,975,390-byte safetensors file plus tokenizer/config;
  package installation is separate.
- Run a two-dialogue dry run first and verify IDs, label mapping, probabilities,
  token counts, hashes, and resume behavior.
- Full scoring appends and fsyncs one record at a time and atomically updates run
  state. Resume skips exact example IDs only after validating their text hash and
  model revision.
- Abort on a model revision/label mismatch, any test-split example, duplicate or
  missing game side, non-finite probability, changed text hash, or dependency
  doctor failure.
- Full command and resume command are logged with the result artifact.

## Post-run interpretation note (2026-08-22; not a gate change)

Every fitted Platt slope was negative. The pre-run sentence that reporting raw
scores means a calibrator “cannot manufacture a discrimination claim” was too
strong: reporting exposes an inversion, but an unconstrained Platt map can and
did reverse the ordering. The frozen numerical gate is unchanged and mechanically
passes. The allowed scientific claim is narrowed in
`v2/results/bridge/README.md` to transferable information in an external score
after target calibration; zero-shot detector success is explicitly rejected.
