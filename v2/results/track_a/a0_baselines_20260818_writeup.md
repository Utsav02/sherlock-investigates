# Track A, arm A0 — lexical and statistical passive baselines

**Date:** 2026-08-17 · **Stage:** B, steps 1 and 3 of `v2/experiment_design.md` §18
**Run:** `a0_baselines_20260818_010835.json` (every number below is from that file)
**Corpus:** Jones & Bergen 2025 **five-minute main study only** — Gate 0
CONDITIONAL, evaluation and local development approved (registry §12). The
15-minute study has no resolved Gate 0 and was not read.

```
canonical source_revision  6891d9a8a2c4        (9 source files, hashes in MANIFEST.json)
canonical dialogues sha256 8c1050dc0f84eda1…   games sha256 807963d4d896ab1b…
frozen split main_study_v1 256543688a5da358…   (unchanged; rebuilt and matched)
```

**The frozen test split was not read.** It is Gate 5, one shot.

---

## 0. Headline, in four lines

1. **A pure-stdlib TF-IDF logistic regression reaches 95.4% game accuracy**
   against a 51.5% majority baseline: **+43.9pp**, participant-clustered 95% CI
   **[+40.4, +47.6]pp**. Every trained A0 detector clears the frozen 10pp target
   by a wide margin.
2. **Template normalisation changes almost nothing** (|Δ| ≤ 1.1pp). The
   artefact alarm the analysis was built to trip **does not fire** — but §6
   explains why that is much weaker evidence than it looks.
3. **Calibration is bad where discrimination is best.** TF-IDF has the lowest
   Brier (0.1175) *and* the worst ECE (0.173); it is systematically
   underconfident. That is where the remaining headroom is, and it is P2's point.
4. **P1's frozen 10pp target is mis-anchored and must be re-frozen before A2
   runs.** It assumed "A0 ≈ 53%". A0 is 95.4%. See §7 — this is the finding with
   the largest consequence for the design.

---

## 1. What was and was not run

| | Status |
|---|---|
| Canonical normalisation of the 5-minute corpus (§9 schema) | **done** — `v2/scripts/build_canonical.py` |
| A0: majority, random, length, punctuation, function words, TF-IDF LR | **done** |
| Contrast **P1** procedure, 11-fold leave-one-component-out over train+dev | **done** |
| Contrast **P2** out-of-fold calibration | **done** |
| Template-normalisation sensitivity | **done**, 3 text variants |
| **A1** (Inverse Turing Bench reproduction) | **deliberately not run** |
| **A2** (classifier head on a frozen temporally clean model) | does not exist |
| Gate 1 rung 2 (SONA ↔ Prolific population holdout) | **not run this session** |
| Gate 1 rungs 4–5 (held-out prompt, held-out family) | **not run this session** |

**P1 as frozen was not evaluated, and this document does not claim it was.**
P1 is *A2 vs A0*. A2 does not exist — the temporally clean checkpoint is still an
open design decision (§19). What ran is the **P1 procedure** with A0's trained
detectors in the estimator slot and A0's trivial baseline in the reference slot.
That fixes the number A2 must beat and exercises the machinery end to end. P2 is
evaluated directly, because calibration is well defined for any estimator that
emits a probability.

**A1 stays unimplemented on purpose.** Its 557 games are 48.9% of the corpus and
biased long (90.5 vs 65.0 mean tokens per released dialogue), and the A2-vs-A1
head-to-head was demoted to interval reporting. It needs its own session with
that coverage bias stated up front, not a line in a table here.

---

## 2. Canonical layer

`v2/data/canonical/main_study_v1/` (gitignored — registry §5 forbids
republishing the text): 2,280 dialogue rows, 1,140 game rows, exactly 1,140
human and 1,140 AI dialogues.

Every row carries the §9 provenance block. `target_origin` is **null** on every
row and that is deliberate: the field records where an SFT *target* came from,
and a canonical conversation has no target. Filling it with a plausible-looking
value would be the §7.1 error in a different layer.

`canonical_policy.check_columns` is called with the columns the loader intends
to **keep**, for every source table it touches. `tt_profile.other` is absent from
every kept list and `tests/test_v2_track_a.py` asserts both that absence and that
the policy still raises if someone adds it back.

Two integrity items from the Stage A inspection were handled and recorded rather
than silently resolved:

- `tt_transcripts` game 2197 duplicates both labels → kept first in file order.
- `tt_verdict` game 2197 has two rows disagreeing on confidence (100 vs 45) →
  kept **verdict id 1401, confidence 100**, by the rule *lowest verdict id*,
  which does not depend on file order. Recorded in `manifest.json`.

Measured, and relevant later: the human witness sat in slot **A in 577 of 1,140**
games. **48** dialogues carry no messages at all — 40 games have at least one
such side and 8 games have two. Counting the *witness* side only, **80** of the
2,280 dialogues have no witness message, which is the count that matters for §6
because it is what a text detector sees as an empty document.

---

## 3. Contrast P1 — paired accuracy, 11-fold leave-one-component-out

911 games (train+dev), 11 folds = the frozen co-occurrence components. Every game
is scored by a model that never saw either of its participants.

| Detector | game acc | vs majority | discordance | McNemar *p* (unclustered) |
|---|---:|---:|---:|---:|
| random | 0.5000 | −1.48pp | 0.515 | — |
| **majority** (always answer slot A) | **0.5148** | — | — | — |
| punctuation | 0.7903 | **+27.55pp** | 0.477 | 5.7e−34 |
| length | 0.8266 | **+31.17pp** | 0.500 | 9.2e−42 |
| function words | 0.8463 | **+33.15pp** | 0.507 | 3.1e−46 |
| **TF-IDF LR** | **0.9539** | **+43.91pp** | 0.500 | 9.7e−91 |

The frozen target is 10pp and the frozen MDD is 5.25pp at ICC 0.10. The weakest
trained detector beats the target by 2.8×.

Two notes on the baselines, because both are easy to get wrong:

- **`majority` is not 50%.** For a forced-choice A/B task the honest majority
  rule is "always answer the training fold's majority human position", which is
  51.48% here. It is applied as an explicit **tie-break**, not by nudging the
  emitted probability — nudging pushed the slot-position prior into dialogue-level
  AUROC (measured 0.5148 instead of 0.5000), reporting a property of the seating
  arrangement as if it were a text signal.
- **The measured 53% in the precision document is the *human interrogator's*
  accuracy**, not a machine baseline. Conflating the two is what produced the
  mis-anchored target in §7.

### The independence unit, and why it barely matters here

Every CI is a percentile cluster bootstrap, 1,000 replicates, seed 20260817, with
one resample shared across detectors and metrics so paired differences come from
the same replicate as the levels.

TF-IDF accuracy difference vs majority, by unit:

| Unit | clusters | diff 95% CI | width |
|---|---:|---|---:|
| game *(naive; anti-conservative)* | 911 | [+40.07, +47.48]pp | 7.41 |
| interrogator | 241 | [+40.44, +47.62]pp | 7.18 |
| human witness | 252 | [+40.37, +47.46]pp | 7.09 |
| **participant** *(reported headline)* | 241 / 252 | **[+40.44, +47.62]pp** | 7.18 |
| component *(valid, only 11 clusters)* | 11 | [+41.36, +49.28]pp | 7.92 |

**The choice of unit moves the interval by well under 1pp.** That is an empirical
confirmation of the precision document's §2(c): with at most 4 games per
interrogator the Kish design effect is small, so *n* binds and the correlation
does not. The naive game-level interval is not materially anti-conservative on
this corpus — which is worth knowing, and is not something that could be assumed
in advance.

Stated as a limitation: games are clustered by interrogator **and** human witness
(crossed, not nested). A proper two-way crossed bootstrap is **not** implemented.
`participant` is the wider of the two marginal cluster bootstraps — a
conservative stand-in that does not model the interaction term. Given the
observed ≤1pp spread across every unit including the fully-valid component
bootstrap, the missing interaction cannot plausibly overturn a 44pp effect, but
it would matter for a contrast near its MDD.

---

## 4. Contrast P2 — out-of-fold calibration

| Detector | dialogue Brier | log loss | **ECE** | game Brier | dialogue AUROC (participant CI) |
|---|---:|---:|---:|---:|---|
| majority / random | 0.2500 | 0.6931 | 0.000 | 0.2500 | 0.500 |
| punctuation | 0.1926 | 0.5779 | 0.091 | 0.1785 | 0.778 [0.752, 0.804] |
| length | 0.1883 | 0.5613 | 0.038 | 0.1574 | 0.789 [0.768, 0.808] |
| function words | 0.1669 | 0.5572 | 0.045 | 0.1292 | 0.835 [0.817, 0.853] |
| **TF-IDF LR** | **0.1175** | **0.3977** | **0.173** | **0.1014** | **0.952 [0.942, 0.961]** |

**The best discriminator is the worst-calibrated estimator, and it is
underconfident in a very regular way.** Its out-of-fold reliability curve:

| predicted bin | n | mean p | observed |
|---|---:|---:|---:|
| [0.1, 0.2) | 179 | 0.158 | **0.006** |
| [0.2, 0.3) | 260 | 0.248 | **0.050** |
| [0.3, 0.4) | 249 | 0.350 | 0.129 |
| [0.4, 0.5) | 243 | 0.441 | 0.333 |
| [0.5, 0.6) | 160 | 0.547 | 0.662 |
| [0.6, 0.7) | 174 | 0.652 | **0.908** |
| [0.7, 0.8) | 243 | 0.749 | **0.967** |
| [0.8, 0.9) | 235 | 0.845 | **0.987** |

Monotone and consistently too timid at both ends — the classic shape a ridge
penalty produces. This is *good news for the design*, not bad: it is exactly the
defect a frozen, out-of-fold-calibrated estimator (design §5, §6.1) exists to
fix, and it says the remaining work on this corpus is calibration, not accuracy.
No calibration layer was fitted here; P2 reports the estimator as-is, which is
what "out-of-fold calibration **of the same estimator**" means.

---

## 5. Template-normalisation sensitivity

Three text variants, identical pipeline, same folds and seed:

- `raw` — witness messages as stored.
- `nostub` — anonymisation placeholders (`<NAME>`, `[LOCATION]`, …) removed.
  These were written by the authors' GPT-4o redaction pass, not by the witness.
- `nostub_nochanged` — additionally drops every message the anonymiser rewrote
  (`is_changed = TRUE`), which contains machine-authored text inside *both*
  classes.

Δ = raw − variant, in game accuracy:

| Detector | raw | Δ vs `nostub` | Δ vs `nostub_nochanged` |
|---|---:|---:|---:|
| length | 0.8266 | +0.0000 | +0.0050 |
| punctuation | 0.7903 | +0.0044 | +0.0016 |
| function words | 0.8463 | −0.0011 | +0.0104 |
| TF-IDF LR | 0.9539 | −0.0011 | +0.0055 |

**Largest movement anywhere: 1.04pp.** Two of the deltas are negative — stripping
the artefact made the detector marginally *better*.

**So the alarm does not fire.** The instruction for this analysis was that a high
lexical score with a large delta is dataset-artefact leakage and should be said
loudly. That combination is not present: the score is high and the delta is ~0.
The placeholder tokens and the anonymiser's rewrites carry essentially none of
the signal.

---

## 6. Why the clean delta does NOT clear this signal — read this before quoting §5

A small normalisation delta licenses exactly one conclusion: *the two artefacts I
was able to strip are not the source of the signal.* It says nothing about the
artefacts I could not strip, and on this corpus those are the big ones.

**The top-weighted TF-IDF terms show the model keying on register, and the
register was assigned by the experiment.** Inspecting the weights (fit on
train+dev, for inspection only, never used for a score):

- toward **AI**: `fr`, `yeah`, `ngl`, `lol`, `u`, `tho`, `mostly`, `gaming`,
  `idk`, `chill`, `kinda`, `nah`
- toward **human**: `i`, `but`, `hi`, `hello`, `yes`, `no`, `my`, `am`, `i_am`,
  `how_are`, `i_have`

The AI witnesses were prompted into a casual young-person persona; the humans,
typing fast under a five-minute clock, wrote `hello` and `i am`. **That contrast
is a property of this study's five prompt templates and its time limit, not a
general human-versus-AI distinction** — and template normalisation as defined
cannot detect it, because the "template" doing the damage is the *system prompt
given to the witness*, not a formatting artefact in the transcript.

Three further axes the normalisation test does not touch:

1. **Model identity is in-distribution.** Folds hold out *people*, not *systems*.
   All six witness systems appear in every training fold, so this is a six-system
   classifier, not an AI detector. The held-out-family rung (Gate 1 rung 5) is the
   test for that and was not run.
2. **Harness asymmetry.** AI witnesses emitted more messages (mean 4.22 vs 3.33)
   and fell silent far less often (22 vs 46 empty witness sides). Per the Stage A
   inspection §10 item 6, AI turns were released with an artificial per-character
   delay, so message count is partly a property of the apparatus. Excluding
   empty-side games barely moves the number (0.9539 → 0.9591), so this is small —
   but it is a real, unremovable component.
3. **Every witness system is easy, so it is not just ELIZA** — and the pattern
   complicates the pure-persona story rather than confirming it:

| Witness system | games | TF-IDF acc |
|---|---:|---:|
| eliza | 116 | 0.987 |
| gpt-4.5_minimal | 124 | 0.960 |
| llama-405b_quinn | 223 | 0.964 |
| gpt-4.5_quinn | 230 | 0.957 |
| gpt-4o_minimal | 112 | 0.924 |
| llama-405b_minimal | 106 | 0.915 |

Excluding ELIZA: 0.949. Persona-only 0.960 vs minimal-only 0.934. **The
no-persona systems are also detected at ~92–96%**, which the slang story alone
does not explain, so register is a major driver but demonstrably not the whole
one. Decomposing it properly is the held-out-prompt rung (rung 4), not run here.

**The honest summary of §5 + §6:** the signal survives the artefacts I could
remove, and remains untested against the ones I could not. A clean delta is a
necessary condition that has been met, not a sufficient one that has been
cleared.

---

## 7. The finding with the largest consequence: P1's target is mis-anchored

`precision_and_contrasts.md` §4 sets P1's 10pp target like this:

> **10pp (C1, C2, C5, C8, C9).** A0 on this corpus is ~53% (measured overall
> interrogator accuracy, 605/1141). A detector worth the Track A machinery should
> clear its baseline by at least 10pp.

**That anchor substitutes the human interrogator's accuracy for A0's.** A0 is not
53%. A0 — a stdlib TF-IDF logistic regression with no pretrained model anywhere
in it — is **95.4%**. The consequence is arithmetic: P1 asks A2 to beat A0 by
10pp, and only **4.6pp of headroom exists**. As frozen, **P1 is unpassable by
construction**, and it would have been read as "A2 adds nothing" when the real
situation is that the baseline saturated the metric.

Recommended, and explicitly **not** applied here because re-freezing a contrast
against the run that motivated it is selection on the evaluation set — the error
this repo has already recorded twice (`t_think_07` regex patching 2026-07-28; the
biased F1 threshold sweep 2026-08-07). The owner should decide, in writing, before
A2 is built:

1. **Re-anchor P1's baseline to the measured A0 (95.4%), not to human accuracy**,
   and set a target A2 could actually clear — or drop accuracy as P1's metric.
2. **Promote calibration to the primary contrast.** Discrimination is saturated;
   ECE 0.173 at AUROC 0.952 is where the headroom is, and design §5 already
   assigns the frozen estimator the job of *owning the probability*, not of
   winning an accuracy race.
3. **Re-anchor against the strongest holdout, not the pooled number.** A0's 95.4%
   is an in-distribution-model figure. The number A2 must beat under held-out
   prompts and held-out families is unknown and is probably much lower.

---

## 8. Is Gate 1 reachable on the cleared corpus alone?

Against the proposed Gate 1 wording in `gate1_out_of_source.md` §5, which requires
rungs 1–3 together:

| Rung | Requirement | Status |
|---|---|---|
| 1 | Held-out people, cross-fitted, out-of-fold calibration | **PASSES.** +43.9pp [+40.4, +47.6]; calibration measured and poor but reportable. |
| 2 | Held-out population, SONA ↔ Prolific, both directions | **NOT RUN** (out of scope this session). Ready: `interrogator_recruitment_source` is in the canonical layer; MDD 6.3 / 7.1pp. |
| 3 | Collection-artifact removal | **PASSES on the artefacts testable here** (Δ ≤ 1.1pp), with the §6 caveat that the dominant source-specific driver is untested. |

**Answer: yes, Gate 1 is reachable on the cleared corpus alone — and it is close
to being reached, but it is not reached by this session.** Rung 2 is the only
required rung outstanding, it needs no new permission, no new data and no GPU,
and against a 44pp effect with a 6.3pp MDD it is very likely to pass. One session
should finish it.

**Three cautions on that "yes", which matter more than the verdict:**

1. **Gate 1 passing is a much weaker statement than it sounds.** It would mean
   *a passive detector discriminates on held-out people and survives artefact
   removal on the five-minute Jones & Bergen corpus.* Rungs 4 and 5 (held-out
   prompt, held-out family) are explicitly non-gating, so a Gate 1 pass is
   compatible with the detector being a six-system classifier that collapses on a
   seventh system. Design §4 already forbids the broader claim; §6 here shows the
   corpus actively invites it.
2. **No dataset-level holdout is available.** The conditional strongest rung (6a-c)
   needs the 15-minute study, whose Gate 0 is unresolved. Until that is answered,
   every downstream claim must state that no dataset-level holdout was performed.
3. **Gate 1 was designed to answer "is there enough signal to justify fine-tuning?"
   It has instead answered a different question: there is so much signal that the
   passive-detection task is nearly solved by a bag of words.** That should
   redirect Track A from "can we detect?" to "can we produce a *calibrated*
   probability, and does anything survive a held-out model family?" — both of
   which are already in the design and neither of which needs a fine-tune.

---

## 9. Limitations of this run, stated rather than buried

- **The folds are severely unbalanced**, inherent to the corpus: held-out
  dialogues per fold range from **2 to 720**. One component holds 39.5% of the
  eval games, and several folds hold a single game. The cross-fit is therefore
  dominated by a handful of large folds; small folds contribute almost nothing.
- **The TF-IDF model did not fully converge.** 10 of 11 folds hit the 250-iteration
  cap rather than the convergence tolerance (objective 0.421–0.444, vocabulary
  3,177–4,802). Numbers are stable across all three text variants and all five
  units, so this is very unlikely to be material, but the run does not license
  the word "converged".
- **Witness side only.** Interrogator turns are excluded from every feature. The
  interrogator is the same person in both conversations of a game, and the
  question is about the writer whose identity is in doubt — but this is a design
  choice, and including interrogator reactions would be a different arm.
- **No calibration layer, by design.** P2 scores the estimators as they come out
  of the cross-fit.
- **Top-weighted terms are published as vocabulary items only** (single tokens and
  bigrams), which are lexical statistics rather than participant utterances, and
  so stay inside registry §5's bar on republishing transcript text. No transcript,
  message, or free-text response is reproduced anywhere in this document.

---

## 10. Reproducing

```bash
venv/bin/python v2/scripts/build_canonical.py     # or: make v2-canonical
venv/bin/python v2/scripts/track_a_a0.py          # or: make v2-track-a0
venv/bin/python -m unittest tests.test_v2_track_a
```

Runtime ~5.5 minutes total on CPU, stdlib only (no numpy, no sklearn, no network,
no model inference). 33 unit tests cover the pure helpers, including AUROC and
McNemar against hand-computed values and the `tt_profile.other` policy assertion.

`build_canonical.py --check` re-derives the canonical digests; `track_a_a0.py` is
deterministic given the frozen split and seed 20260817. Verified: the `raw`
variant's point estimates reproduced identically across three separate
invocations (TF-IDF 0.9539 / 0.9515 / 0.1175 each time). Interval widths depend
on the replicate count and are reported at 1,000.
