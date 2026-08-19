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
**Missing (see queue item 5): the literature matrix and the source-coverage
table** required by §18 Stage A steps 1 and 5. One corpus was found and work
accelerated into analysis before the broader prior-art and dataset comparison
was finished. This is a milder replay of V1's failure mode and should be closed
before Stage C is trusted.

**Stage B (Track A, real-passive) — COMPLETE, with a negative generalization
result.** Four sessions, six commits, all artifacts in `v2/results/track_a/`.

| arm | status |
|---|---|
| A0 majority/random/lexical/statistical | done |
| A1 Inverse Turing Bench reproduction | **SKIPPED** — never implemented; see below |
| A2 frozen temporally clean representation + head | done, one run |
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

⚠️ **It is NOT proof that the corpus contains no learnable general signal.**
Earlier wording in `v2/DECISIONS.md` and the A2 writeup says "the ceiling is the
corpus's" — that converts a two-method result into an impossibility claim and is
**queued for correction (item 3)**. Only one neural representation was tested,
with crude mean pooling and a random projection; A1 and A3 were never run.

---

## Reconciliation queue — do these BEFORE any new experiment

1. ~~Retire the V1 "scale Claude scenarios" instruction from the active status.~~ **DONE 2026-08-18.**
2. ~~Fix or isolate the broken V1 edit.~~ **DONE 2026-08-18** — `reverse_scenarios.py`
   had a stray `"""` at line 639 prematurely closing a docstring (the real close
   is line 647), left by an interrupted edit on 2026-08-17. Replaced with a blank
   line; one-line change, no logic touched. Full suite went from *228 tests,
   2 import errors* to **303 tests, OK**. **The fix is in the working tree and is
   deliberately NOT committed** — that file carries ~62 lines of another
   session's uncommitted work, and committing it would sweep those in. Whoever
   owns those edits should commit them together.
3. **Narrow the "corpus ceiling" claim** in `v2/DECISIONS.md` (2026-08-18 final
   entry) and `v2/results/track_a/a2_frozen_rep_20260818_writeup.md` to the
   two-method wording above. `results/` is append-only → correction entry, not an
   edit.
4. **Stop calling A2 "calibrated", or make it so.** `track_a_a2.py` fits ordinary
   L2 logistic regression and *evaluates* Brier/ECE out of fold. It does **not**
   fit a Platt/isotonic/temperature calibrator nested inside the training folds.
   Out-of-fold calibration *evaluation* ≠ out-of-fold *calibration*. Either rename
   to "probabilistic head evaluated for calibration" or add a nested calibration
   stage. The reported ECE/Brier numbers are legitimate either way.
5. **Finish Stage A**: the literature matrix (§18 step 1) and the source-coverage
   table (§18 step 5).
6. **Record the plan revision explicitly.** Track A stopped early relative to the
   frozen plan, which still lists A1, A3, and "out-of-source discrimination and
   calibration" as Gate 1. The clean statement is: *broad Gate 1 failed or
   unavailable; within-configuration passive detection established; A1 skipped as
   a consequence of the stopping decision; A3 deferred to optional Stage D.*
   Note the internal contradiction to resolve: `v2/DECISIONS.md` records
   persona/family rungs as **non-gating**, and the next entry then uses persona
   failure to conclude Track A. Stopping is defensible — but as a recorded plan
   revision, not silently.
7. **Back up the work.** Branch `track-a-ablation` is **9 commits ahead of
   `origin/main` with no remote branch**. Local-only, against the repo's own
   durability rule. Push after a secrets check. *Owner decision — not pushed
   without being asked.*

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
