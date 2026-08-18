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
