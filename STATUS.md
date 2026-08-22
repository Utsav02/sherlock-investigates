# STATUS — sherlock-investigates
**Updated: 2026-08-18** · **Active project: v2** (`v2/experiment_design.md`, revision v2.1)

---

## ⛔ READ THIS BEFORE RESUMING ANYTHING

**The active project is v2. V1 is PAUSED. Do not resume V1 scenario generation.**

Until 2026-08-18 this file ended with a "NEXT SESSION" block instructing the next
session to *scale Claude scenarios to ~300 and assemble the SFT set*. That
instruction is **retired**. It belongs to V1 and it contradicts v2.1, which
defers datasets D1–D5 and does not authorise D5 at all (`v2/experiment_design.md`
§10, §13). Following it would have restarted the abandoned V1 pipeline.

The V1 status is archived verbatim at [`docs/STATUS_v1_archive.md`](docs/STATUS_v1_archive.md).
Its measurements and retractions are still valid history and are cited by the
Decision Logs; its *instructions* are not.

**Two decision logs exist and both are load-bearing:**
- `CLAUDE.md` — V1 entries through 2026-08-17.
- `v2/DECISIONS.md` — v2 Stage A/B entries. Merge into `CLAUDE.md` in date order
  once the tree is clean; until then append there, not to `CLAUDE.md`.

---

## Where the project actually is

**Stage A (audit) — mostly done, two artifacts missing.**
Done: licence/provenance registry, source download with per-file hashes, schema
inspection of both studies, PII policy enforced in code, frozen participant
split, precision/MDD analysis with a frozen contrast set, ITB length-unit
determination.
**Added 2026-08-18** (late, after Stage B — the ordering error is real and is
recorded): `v2/results/stage_a/source_coverage.md` and
`v2/results/stage_a/literature_matrix.md`.
- The **source-coverage table is complete.** Headline: *no existing source
  supplies executable respondents, turn-level beliefs, or verified SFT actions.*
  D0 is the only route to all three, which is independent evidence that Stage C
  is the right next move.
- The **literature matrix is built and verified** against 15 fetched sources
  (2026-08-18). Cells marked [F] full-text, [A] abstract, [R] repo, [U] unverified.
  Two findings changed the record — see below. Remaining: an ITB dataset registry
  record (Gate 0 requirement), a full-text pass on the Group 2 papers before arm
  B2 is committed, and Sleeper Agents/SPADE/HC3/HANSEN (low priority).

**Stage B (Track A, real-passive) — CONCLUDED EARLY, with a negative
generalization result.** Not "complete": two of the four planned arms were never
implemented. Artifacts in `v2/results/track_a/`; claim-level corrections in
`corrections_20260818_213000.md`; plan revision recorded in `v2/DECISIONS.md`
(2026-08-18).

| arm | status |
|---|---|
| A0 majority/random/lexical/statistical | done |
| A1 Inverse Turing Bench reproduction | **SKIPPED** — never implemented; see below |
| A2 frozen clean representation + **probabilistic** head | done, one run — **not calibrated**, see below |
| A3 turn-wise evaluation | **DEFERRED** — never implemented; see below |

**Stage C (Track B, synthetic-active) — NOT STARTED.** This is the next
experimental work, and only after the reconciliation queue below.

---

## What Track A established (use this wording, not "corpus ceiling")

**Established.** On the five-minute Jones & Bergen three-party corpus, passive
human-vs-AI discrimination against *seen* respondent configurations is real and
well characterised: A0 (TF-IDF) 0.960, A2 (frozen Qwen2.5-7B + logistic head)
0.866 [0.842, 0.889] paired accuracy, A2 with out-of-fold ECE 0.101 and genuine
two-sided recognition (74.2% of AI dialogues flagged, 80.4% of humans cleared).
The signal is not an artifact of interrogator text, reply rate/length, redaction
placeholders, or respondent silence — each measured and excluded — and it
transfers between recruitment populations.

**Not established.** Neither estimator survives a **held-out persona prompt**.
Trained on minimal-prompt witnesses and evaluated on persona-prompt witnesses,
A0 scores 0.604 and A2 0.493; reversed, 0.495 and 0.448. All four intervals
contain 0.5, balanced accuracy is 0.47–0.52, and the held-out AI is flagged as AI
in only 5.7–14.6% of games. Under persona holdout both estimators become
*human-recognizers* that label nearly everything human.

**The supported conclusion, stated at the right width:** *two methods — a
bag-of-words baseline and one frozen-representation head — both fail to transfer
between these two prompt families, and the neural one fails harder.* That is
strong evidence of severe prompt dependence in this corpus.

⚠️ **It is NOT proof that the corpus contains no learnable general signal.** The
earlier "the ceiling is the corpus's" wording converted a two-method result into
an impossibility claim; **corrected 2026-08-18** (`v2/DECISIONS.md`;
`corrections_20260818_213000.md` §C1). Only one neural representation was tested,
with crude mean pooling and a 512-d random projection, and A1 and A3 were never
run.

### ⚠️ Two literature findings that change the record (2026-08-18)

**1. Inverse Turing Bench independently replicates the persona collapse — and
bounds our explanation.** ITB evaluates zero-shot judges on this same corpus:
Claude Opus-4.6 and GPT-5.5 fall from ~95.8%/93.4% to ~59.0%/57.6% when the AI
witness has a persona prompt, and human judges from 68.5% to 39.9%. Same
direction and magnitude as our A2 (0.866 → 0.49/0.45), by a different method —
**our finding is not a pipeline artifact.**

But in the same sentence: **GPTZero's performance "remains consistent"** across
the persona split. GPTZero is trained on generic AI text, *not* on this corpus.
That is positive evidence a **persona-invariant signal exists in these
transcripts** which our estimators failed to learn. The distinguishing axis is
not token-vs-semantic (our A0 is token-based and collapsed) but **trained
in-corpus vs trained out-of-corpus**.

**So "the ceiling is the corpus's" is now positively unlikely, not just
unproven.** The Gate 1 verdict is unchanged — a persona-robust *generic* detector
is not a calibrated investigator, which is what v2.1 needs — but the explanation
on the record was wrong and is corrected in `v2/DECISIONS.md`. If Track A is ever
revived, this names the experiment: train out-of-corpus, evaluate in-corpus.

**2. Stage C's named active baseline is out of date.** §13.2 names UoT alone as
arm B2. **BED-LLM** (ICLR 2025) does the same job more rigorously and reports far
larger gains (GPT-4o 93% vs 45% prompt-only, against UoT's +38.1% over direct
prompting); **CA-BED** (2026) extends it to ambiguous/partially-informative
answers, which is the D0 setting; **ClarQ-LLM** supplies an executable provider
agent — requirement R4, which no real corpus in our candidate list provides.
**Recommendation: B2 should be BED-LLM-style EIG with UoT secondary.** Naming the
weaker of two published methods makes Gate 2A easier to pass and less meaningful.
This is a design-doc change and is the **owner's call**, flagged not made.

**Gate 1 verdict, in the form to quote:** *broad Gate 1 failed or unavailable —
failed on the persona rung, unavailable because no out-of-source corpus was ever
readable (the 15-minute study's Gate 0 is unresolved); within-configuration
passive detection established; A1 skipped; A3 deferred.* The stop was a
**judgement call informed by a non-gating rung, not a gate firing** — persona and
family rungs are explicitly non-gating, and that contradiction is resolved in
`v2/DECISIONS.md` (2026-08-18 plan revision).

**A2 is a probabilistic head, not a calibrated one.** It fits ordinary L2 logistic
regression and *measures* Brier/ECE out of fold; there is no nested
Platt/isotonic/temperature stage. Out-of-fold calibration *evaluation* is not
out-of-fold *calibration*. The numbers stand; the label was wrong and is
**corrected 2026-08-18** (§C2). Design §13.1 asks for a calibrated head, so the
arm as built is a recorded deviation.

---

## Reconciliation queue — do these BEFORE any new experiment

1. ~~Retire the V1 "scale Claude scenarios" instruction from the active status.~~ **DONE 2026-08-18.**
2. ~~Isolate the broken V1 edit.~~ **UNBLOCKED 2026-08-18, but NOT DURABLE — still
   needs an owner.** `reverse_scenarios.py` had a stray `"""` at line 639
   prematurely closing a docstring (the real close is line 647), left by an
   interrupted edit on 2026-08-17. Replaced with a blank line; one line, no logic
   touched. Full suite went from *228 tests, 2 import errors* to **303 tests, OK**.

   **Why this is not finished.** Verified: `HEAD`'s copy of the file parses
   cleanly, so the syntax error did not exist in the repository — it arose
   *inside* an uncommitted V1 patch (152 insertions across
   `scripts/data_prep/reverse_scenarios.py` and `tests/test_reverse_scenarios.py`).
   A one-line repair therefore **cannot be committed as an independent fix**:
   staging that file would sweep in the whole patch, and the fix is meaningless
   without it. It survives only as an uncommitted working-tree edit and will be
   lost if the patch is reverted — which is harmless, because reverting also
   removes the error.

   **Owner action required:** review the whole V1 patch and either commit it (with
   this repair folded in) or discard it. Until then the green suite depends on an
   uncommitted edit, which is not a durable state.
   A copy of the pre-repair file is in this session's scratchpad.
3. ~~**Narrow the "corpus ceiling" claim.**~~ **DONE 2026-08-18** — correction
   entry in `v2/DECISIONS.md`, sheet §C1, code docstring fixed. Writeups not
   edited (append-only); the sheet supersedes the claim.
4. ~~**Stop calling A2 "calibrated", or make it so.**~~ **DONE 2026-08-18** —
   renamed to "probabilistic head, evaluated for calibration" in code and in the
   sheet §C2. **No calibrator was built**, deliberately: that is new experimental
   work on a stopped track. Adding a properly nested one (fitted on training folds
   only) remains available if A2 is ever revived.
5. ~~**Finish Stage A**: literature matrix + source-coverage table.~~ **DONE
   2026-08-18** — both artifacts written; 15 sources fetched and verified. Two
   findings changed the record (below). Left open, deliberately: an ITB **dataset**
   registry record (Gate 0 requirement, cheap), a full-text pass on BED-LLM/CA-BED
   before arm B2 is committed, and four low-priority sources.
6. ~~**Record the plan revision explicitly.**~~ **DONE 2026-08-18** — plan-revision
   entry in `v2/DECISIONS.md` and sheet §§C3–C4. A1 SKIPPED, A3 DEFERRED, Stage B
   "concluded early" not complete, Gate 1 restated as failed-or-unavailable, and
   the non-gating contradiction resolved (stop = judgement call, not a gate
   firing).
7. **Back up the work.** Branch `track-a-ablation` has no remote branch and is
   local-only, against the repo's own durability rule. Check the gap with
   `git log --oneline origin/main..HEAD | wc -l` rather than trusting a number
   written here — it goes stale on the next commit. Push after a secrets check.
   *Owner decision — not pushed without being asked.*

---

## Then, and only then: Stage C

Build **D0 only** — the synthetic simulator with known response distributions —
and the **no-training Gate 2A baselines** (random, fixed order, UoT-style
expected-information-gain).

**No D0 SFT until Gate 2A passes.** Gate 2A has no post-hoc failure exception:
a pre-registered non-trained active policy must beat both random and fixed
baselines, or D0 SFT does not proceed (`v2/experiment_design.md` §16).

Do **not** resume V1, and do not add Track A arms — Track A's stopping decision
stands unless item 6 revises it.

---

## Reproducing Track A end to end

```bash
make v2-canonical            # normalize the cleared 5-minute study
make v2-track-a0             # A0 baselines, contrasts P1 + P2
make v2-track-a0-ablation    # witness-only / length-free / capped cells
make v2-rung2                # SONA <-> Prolific transfer
make v2-rung4                # leave-one-system-out / leave-one-persona-out
make v2-a2                   # needs a llama-server with --embeddings; see writeup
venv/bin/python -m unittest discover -s tests     # 303 tests
```

**Test split is UNTOUCHED and must stay so** — it is Gate 5, one shot, 229 games.

## Invariants — do not re-litigate

- `tt_profile.other` never enters a derived artifact; `canonical_policy` fails closed.
- Source transcripts and canonical text are gitignored; only hashes/provenance are published.
- `v2/results/` is append-only — corrections are new timestamped files, never edits.
- Splitting unit = connected component of the participant co-occurrence graph;
  inference clusters on the participant. Different questions, different units.
- The 15-minute study has **no resolved Gate 0** — not read, not evaluated.
- Never `git add -A`; this tree is co-mingled with another session's V1 work.

---

# Active correction programme — 2026-08-22

**Updated:** 2026-08-22 11:20 PDT

## Goal

Correct the V1 repeated-measures analysis and Track A crossed-participant
inference, decide whether the deliverable is a methodological failure study or a
real active-investigation programme, and only then build the smallest justified
bridge experiment. Stage C is paused. The untouched Track A test split remains
untouched, and the unrelated uncommitted V1 scenario patch is out of scope.

## Stages

- [x] 1. Audit raw observations and freeze a correction specification — commit
  `3c33064`; historical V1 prompt outcomes are not recoverable from committed
  aggregates.
- [x] 2. Implement corrected V1 repeated-measures and Track A crossed-participant
  inference — historical p-values withdrawn; future prompt outcomes persisted;
  dyadic/component intervals implemented; corrected A0/A2 artifacts generated;
  304 tests green.
- [x] 3. Explain the deliverable fork and record the owner's decision —
  **HYBRID:** finish an honest methodological failure study, explicitly including
  poor planning/ordering as a cause, then continue the real-active programme
  behind the bridge gates.
- [ ] 4. Implement the selected smallest bridge experiment ← **IN PROGRESS:**
  select and provenance-lock an out-of-corpus detector that can be executed
  reproducibly, add nested calibration, and freeze D0's mandatory real-replay
  criterion before any Stage C implementation.
- [ ] 5. Verify the complete record, update the public README/status, and commit
  each clean stage by exact path. No push without owner authorization.

## Resume

```bash
git status --short
rg -n 'prompt|opener|closure|outputs|rows' results/analysis/*.json scripts/eval/dose_curve.py
rg -n 'bootstrap_intervals|cluster_units|widen' v2/scripts/track_a_a0.py
venv/bin/python -m unittest discover -s tests
```

Resume at Stage 4. Do not begin Stage C or read the frozen Track A test split.

## Decisions made

- Stage C is paused pending inference corrections and an explicit deliverable
  choice.
- Correct both V1 repeated-measures inference and Track A crossed-participant
  intervals before making new claims.
- Ask the owner to choose the deliverable after explaining concrete options.
- Deliverable: **hybrid**. Package the methodological failure study first, then
  continue the active-investigation programme as a gated extension.
- Failure framing: distinguish genuine negative scientific results from failures
  caused by poor planning, premature implementation, construct mismatch, and
  invalid inference. Record both plainly.
- V1 correction depth: withdraw unsupported Fisher/Wilson pooled inference and
  mark the historical dose curves descriptive; do not rerun GPU experiments.
  Upgrade logging so any future run retains prompt-level outcomes.
- Git: owner authorised path-specific commits alongside the dirty tree. Never
  stage the unrelated scenario patch, generated datasets, draft files, or HTML.
