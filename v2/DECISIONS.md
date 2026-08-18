# v2 Decision Log

Append-only, same format and rules as the Decision Log in the root `CLAUDE.md`.

**Why this file exists separately.** These entries were written while another
session held uncommitted edits in the root `CLAUDE.md` and in
`scripts/data_prep/reverse_scenarios.py`. Appending here avoids a conflicting
write to a file this session does not own. **Merge these entries into
`CLAUDE.md`'s Decision Log, in date order, once the working tree is clean**, then
leave this file as a pointer rather than deleting it.

Entries for 2026-08-17 (v2.1 design revision, Stage A execution) are already in
`CLAUDE.md`. This file starts at Stage B.

---

### 2026-08-18 — Canonical layer built for the cleared 5-minute study; PII policy is enforced in code, not prose

**Decision:** `v2/scripts/build_canonical.py` normalizes the 5-minute
three-party study into `v2/data/canonical/`. `canonical_policy.check_columns` is
called with the columns the loader keeps, per table; `tt_profile.other` is absent
from every keep-list and a test asserts the policy still raises if it is
re-added. Game 2197's duplicate verdict rows are resolved by lowest id (keeping
confidence 100, dropping 45), recorded in the manifest. `target_origin` is `null`
on every canonical row.

**Reasoning:** The PII decision has to bind future code, not just future readers.
A Markdown note saying "we excluded `tt_profile.other`" is exactly the kind of
control that the four lost v1 runs proved unreliable — it depends on someone
remembering. A fail-closed check in the loader means a later session that adds
the column back gets a test failure rather than a silent re-introduction of a
live contact channel into a derived artefact.

The exposure was subsequently bounded by a full sweep of every free-text surface
in both studies (six `tt_profile` free-text columns, `tt_verdict.reason`, and all
36,991 message bodies), scanned for email / 24-hex / URL / phone patterns with
counts reported and no values printed. Hits: `tt_profile.other` only — one 24-hex
token in the 5-minute study, one email plus one 24-hex token in the 15-minute
study. Everything else clean. **So excluding this single column is sufficient;
the policy does not need widening.** That is what the disclosure draft to the
authors should state, because it tells them exactly how far the problem extends.

`target_origin` is null because a canonical conversation has no SFT target.
Filling it with a plausible value would be the §7.1 "null, not a plausible
untrained number" error committed one layer down, and §9 explicitly forbids
conflating canonical observations with SFT targets.

**Alternatives considered:** Excluding `tt_profile.other` at read time only —
rejected; the exclusion has to survive a new loader written by a future session.
Widening the exclusion to all free-text columns pre-emptively — rejected once the
sweep bounded the exposure; discarding usable strategy/emotion free text on
suspicion would cost real signal for no measured risk. Dropping game 2197
entirely — rejected; lowest-id resolution is deterministic, recorded, and keeps
the game, and the duplicate is independently useful as the explanation for the
556-vs-557 discrepancy in the Inverse Turing Bench length reconstruction.

---

### 2026-08-18 — A0 results are PROVISIONAL: the transcript column contains the interrogator's turns, and length encodes typing speed

**Decision:** The Track A arm A0 numbers (TF-IDF LR 0.954, function words 0.846,
length 0.827, punctuation 0.790) are recorded but **must not be read as passing
Gate 1 rung 1** until two ablations run: (1) witness turns only, and (2) witness
turns only with every length-derived feature removed. The three-way table is
reported before any Gate 1 claim. The `a0_baselines_20260818_*` artefacts stay in
place unedited; this entry is the caveat, not a retraction.

**Reasoning:** Two measured artefacts, neither caught by the template-
normalization check (whose largest delta anywhere was 1.04pp, with two deltas
negative):

1. **The `transcript` column interleaves both sides.** Parsing line prefixes
   returns `Counter({'I': 17, 'W': 17})` on the first row. Unless the loader
   strips `I:` lines, roughly half the classified text is the interrogator, not
   the respondent. RQ1 asks whether a detector distinguishes human from AI
   *respondents*; a classifier reading the interrogator's side answers a
   different question. The direct label-leak through interrogator suspicion is
   weak — bot/AI/robot mentions appear in 163/1140 (14.3%) of human-witness
   conversations vs 186/1140 (16.3%) of AI-witness ones — so this is confounding
   and dilution rather than a smoking gun, but the direction cannot be known
   without the ablation.

2. **Length is a rate artefact, not a text property.** Median witness words:
   human 24, AI 31. Median *interrogator* words: 20 in human conversations, 26 in
   AI conversations. Median messages per conversation: 6 vs 8. Nothing about the
   interrogator changed between the two conversations of a game — the same person
   ran both, simultaneously, under one 5-minute clock. The AI replies fast
   (`secs_per_char=0.3` for all twelve configs) so more exchanges fit. `max_length`
   is uniformly 1000, so this is latency, not a generation cap. Message count and
   total length therefore measure **typing speed**, a property of the apparatus
   that no deployed detector has. Template normalization structurally cannot
   catch it: it is not a template or a redaction placeholder, it is a rate.

**Why this outranks the headline number.** On dialogue from this same corpus,
Inverse Turing Bench reports GPTZero at 89.4% and Claude Opus 4.6 at 77.9%. A
stdlib TF-IDF logistic regression at 95.4% beats both. The tasks are not
identical — ITB is paired discrimination, A0 is single-transcript classification
— so the ordering is not impossible. But a from-scratch bag-of-words model
outscoring a commercial detector and a frontier LLM is the exact shape of the
2026-08-07 v1 finding, where a BoW baseline matching a sophisticated method was
the diagnostic that the instrument was reading an artefact rather than the
concept. Given human 24 vs AI 31 median words, a length-only feature reaching
0.827 is more separation than that gap alone supports, which points at message
count and per-message structure — i.e. mostly the rate artefact.

The session's own persona-register finding (top TF-IDF weights `fr, ngl, lol, u,
tho, idk` toward AI vs `hello, i am, yes, no` toward human, tracking the witness
system prompt) is correct and complementary. Folds hold out people, not systems,
so A0 is a six-system classifier; that no-persona systems still score 0.92–0.96
means register is not the whole driver, and decomposing it remains rung 4.

**Alternatives considered:** Reporting 0.954 as Gate 1 rung 1 passing "decisively"
— rejected; that reports a number as evidence for a claim the measurement does
not support, which is the verification rule this repo exists to enforce.
Retracting the A0 artefacts — rejected; the run is real, correctly executed
against its own specification, and the three-way ablation table is only
interpretable beside it. Treating the register finding as sufficient explanation
— rejected; register is one channel, the interrogator turns and the rate artefact
are two more, and only the ablation separates them.

---

### 2026-08-18 — P1 re-anchored from discrimination to calibration

**Decision:** Contrast P1 changes from "A2 vs A0 on accuracy, 10pp target" to
**A2 vs A0 on calibration (ECE and Brier, out of fold)**, with discrimination
demoted to a guardrail: A2 must not degrade accuracy by more than a
pre-registered margin. The frozen 10pp accuracy target is retired. P2 and P3 are
unchanged. This is a deviation from a frozen contrast and is labelled as one.

**Reasoning:** The frozen target was mis-anchored on "A0 ≈ 53%", but 53% is the
**human interrogator's** accuracy in the published study, not a detector
baseline. That is a category error: it set an automated detector's floor using a
number describing people. Measured A0 is 95.4%, leaving ≤4.6pp of headroom, so
P1 as frozen was unpassable by construction and would have reported "A2 adds
nothing" for arithmetic reasons rather than scientific ones.

Re-anchoring after seeing a run is normally selection on the evaluation set — the
`t_think_07` failure — so the justification must not depend on the observed
direction, and here it does not:

1. **Structural ceiling.** A contrast with ≤4.6pp of headroom cannot resolve a
   real effect whatever threshold is chosen. This follows from A0's ceiling
   alone, not from which arm won.
2. **External anchor.** Any retained discrimination target is anchored on
   published outside numbers (ITB: GPTZero 89.4%, Claude Opus 4.6 77.9%), not on
   our own A0. An outside reference is not selection on our evaluation set.
3. **It moves toward the stated contribution, not toward the winner.** v2.1's
   named centre is calibration-aware investigation. P2 already found that the
   best discriminator is the worst calibrated — TF-IDF Brier 0.1175 but ECE
   0.173, monotonically underconfident (predicted 0.75 → observed 0.97). That gap
   is wide open, and ITB does not report calibration at all.

The session that ran A0 flagged the mis-anchoring and explicitly declined to
re-freeze against its own motivating run. That was correct and is the reason this
decision is recorded here with its reasoning attached rather than applied
silently in code.

**Alternatives considered:** Re-freezing the accuracy target at a level A2 could
plausibly clear — rejected outright; that is fitting the gate to the result.
Keeping the 10pp target and reporting the inevitable failure — rejected; a gate
known in advance to be unpassable for arithmetic reasons produces a misleading
negative, which is worse than no gate. Dropping P1 entirely and relying on P2 —
rejected; P2 is out-of-fold calibration of a single arm, while the project needs
a *contrast* between arms, which is what re-anchored P1 supplies.

**Consequence for the empirical record:** because the A0 numbers that exposed the
mis-anchoring are themselves provisional (previous entry), the ceiling argument
must be re-checked after the witness-only, length-free ablation. If ablated A0
falls well below 95%, the headroom argument weakens and P1's anchoring should be
revisited **once**, in writing, before any A2 run — not after.

---

### 2026-08-18 — PRE-REGISTERED floor for the A0 ablation (written BEFORE the run)

**Decision:** Before proceeding from Task 2 to Gate 1 rung 2, condition
**A0-wit-nolen** (witness turns only, every length-derived feature removed, raw
text variant) must clear this floor:

> The best-performing detector's **game accuracy participant-clustered 95% CI
> lower bound must exceed 0.65.**

If the lower bound is at or below 0.65, the ablation has failed, Gate 1 rung 1 has
NOT passed, and **rung 2 is not run in this session.**

**Reasoning, and every input to it is known without seeing the result:**

- The majority baseline on this corpus is **0.5148** (always answer the training
  fold's majority human slot). A floor of 0.65 is **+13.5pp** over that baseline.
- The frozen MDD for this contrast is **5.25pp** at ICC 0.10
  (`precision_and_contrasts.md` §5, contrast P1/C8). 13.5pp is 2.6× the MDD, so
  clearing the floor cannot be a sampling artefact.
- `precision_and_contrasts.md` §4 set "worth the Track A machinery" at **≥10pp
  over baseline**. 13.5pp clears that bar with margin, so the floor inherits an
  externally-fixed justification rather than one invented for this run.
- The bound is on the **CI lower bound, not the point estimate**, so a detector
  that clears it by luck on a favourable resample does not pass.
- 0.65 sits far below the observed un-ablated 0.954 and far above 0.5148, so it
  presupposes neither outcome. It is a floor for "there is real discrimination
  left after the artefacts are removed", not a floor for "the headline survived".

**Why a pre-registered floor at all.** The two artefacts under test (interrogator
turns in the classified text; length as a proxy for reply rate) were found in
review *after* the headline was reported. Choosing the pass mark after seeing the
ablated number would be selection on the evaluation set — the `t_think_07` regex
failure (2026-07-28) and the biased F1 threshold sweep (2026-08-07) are the two
prior instances in this repo, and both are cited in the frozen contrast document
as the reason not to re-split or re-tune against a motivating run.

**Alternatives considered:** A floor on the point estimate — rejected; weaker, and
the CI is already computed. A floor of 0.90 ("must stay near the headline") —
rejected as far too strict: the question is whether real signal survives, not
whether the artefact-inflated number survives, and 0.90 would fail a genuinely
useful detector. A floor at the 0.5148 baseline plus the 5.25pp MDD (0.567) —
rejected as too weak: statistically distinguishable from chance is a much lower
bar than "useful discrimination", which is what Gate 1 asks for.

---

### 2026-08-18 — Ablation result: the signal is lexical, not rate; rung 2 transfers; the PROVISIONAL flag is lifted with two corrections

**Decision:** The A0 numbers are no longer provisional. The three-way ablation
ran (plus two added cells), the pre-registered floor was cleared, and Gate 1
rung 2 ran and passed. **Gate 1 rung 1 stands**, on a corrected basis. Full
numbers: `v2/results/track_a/a0_ablation_20260818_writeup.md`. Two corrections
attach to the earlier artefact, which is left unedited:

1. The `length` family's 0.827 is **not a detector result** and must never be
   cited as evidence of detection. It is a measurement of reply-rate asymmetry.
   Under ablation all ten of its features are length-derived, so the family is
   empty and degenerates to the class prior (0.500).
2. Rung 1's verdict now rests on the length-free number — 0.9539 (with the no-op
   caveat below) or 0.9210 under the strict length-equalised control — not on the
   un-ablated 0.954.

**Reasoning:**

*Artefact (a) was real about the release and absent from this pipeline.* The
`transcript` column does interleave both sides (`Counter({'I': 17, 'W': 17})` on
row 1; 9,277 I / 8,811 W across all 2,282 rows). But `build_canonical.py` builds
from `tt_message_anonymized.sender_role` and never reads that column. Verified by
reconstructing witness-only text independently from the source and comparing:
**0 of 2,280 dialogues differed.** `A0-full` ≡ `A0-witness` on every metric is
the empirical confirmation. The cell was run rather than skipped precisely
because "the loader already does the right thing" is the kind of claim that
should be measured, not asserted.

*Artefact (b) was real and is confirmed in full,* including the decisive detail:
the **interrogator** writes 20 words opposite a human and 26 opposite an AI,
while being the same person running both conversations simultaneously under one
5-minute clock. A difference in their own output can only come from how fast the
other side replied. `max_length` is 1000 and `secs_per_char` 0.3 across all
twelve configs, so it is latency, not a cap.

*Why an unrequested fifth cell was added, and it changed the conclusion's basis.*
`drop_length` is a **no-op for TF-IDF** — it has no explicit length features — so
the identical 0.9539 in the `nolen` cell is the same model, not a survivor.
Its vectors were already L2-normalised (verified: all norms exactly 1.000000000),
closing the vector-norm channel the task flagged, but a longer document still
activates more distinct terms, and nothing in the requested design reaches that.
The added `A0-wit-nolen-capped` cell truncates every witness side to a fixed
20-token budget, equalising median witness words from 24/30 to **20/20**.
TF-IDF holds at **0.9210 [0.9032, 0.9383]**. Without this cell the headline claim
would have rested on a no-op, which is exactly the shape of error this repo keeps
finding. The drop is an **upper bound** on the length channel, since truncation
also removes information.

*The signal is lexical.* Function words (pure rates, TTR dropped as a length
proxy) hold 0.8260, and 0.8019 on 20-token snippets; punctuation rates hold
0.7827 / 0.7124. A model seeing twenty words per side with all length cues
removed picks the human 92% of the time.

*Rung 2 transfers.* Floor cleared (TF-IDF CI lower bound 0.9394; best genuinely
ablated family, function words, 0.8035 — both above the pre-registered 0.65), so
rung 2 ran. A0-wit-nolen transfers at 0.9250 (SONA→Prolific) and 0.9540
(Prolific→SONA) against 0.9539 in-source. Recorded as a **between-experiment
holdout confounded with batch/lobby structure**, never as a clean population
holdout: recruitment source is perfectly nested in the components (0 of 15
mixed), so "different people" and "different batch" cannot be separated. All six
witness systems are present in both sources (the task brief said four; measured
six), Prolific share 0.54–0.59 per system, so imbalance does not explain the
transfer. Incidental: the majority baseline drops *below* chance in transfer
(0.4692 / 0.4936), confirming the halves are separately batched.

**Consequence for P1, discharging the required re-check.** The previous entry
required the ceiling argument be re-checked once, in writing, before any A2 run.
Done: A0 is 0.9539 un-ablated, 0.9539 length-free, and 0.9210 under the strictest
control — headroom to a perfect score of 4.6pp, 4.6pp and **7.9pp**, all below
the frozen 10pp target. **The ceiling argument survives on every variant, so the
re-anchoring from discrimination to calibration stands and is not revisited.**
Calibration remains the open axis: TF-IDF ECE 0.173 / 0.141, function words
0.045–0.063.

**What is still NOT established, and it is the load-bearing gap:** rung 4. Folds
hold out people, not systems, so every number here is in-distribution across the
six witnesses. A0 beats ITB's published GPTZero (89.4%) and Claude Opus 4.6
(77.9%) on related data, and the per-system table shows why that must not be
published before the witness-system decomposition: the no-persona systems are
already weakest and degrade most under transfer (llama-405b_minimal 0.915 →
0.828). Until rung 4 runs, the honest description is "distinguishes these six
witnesses from these humans", not "detects AI".

**Alternatives considered:** Reporting the `nolen` TF-IDF 0.9539 as "survived
length removal" — rejected; it is a no-op for that detector and reporting it that
way would have been a false ablation claim. Skipping the `A0-witness` cell as
redundant once the loader was shown correct — rejected; running it converts an
assertion into a measurement for one wasted minute. Treating the capped cell as
the new headline — rejected; truncation removes information as well as length, so
it is a control and a lower bound, not the primary estimate. Proceeding to rung 4
in this session — rejected; it is a different holdout with its own precision
question and belongs in its own session, as the A1 decision already established.
