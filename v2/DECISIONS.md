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

---

### 2026-08-18 — CORRECTION: the "separately batched" inference is retracted; the cause is slot-position noise

**Decision:** The claim in the previous entry — *"the majority baseline drops
below chance in transfer (0.4692 / 0.4936), confirming the halves are separately
batched"* — is **withdrawn**. It was unsupported. The correct mechanism is
sampling noise in which conversation slot held the human, and it reproduces the
observed numbers exactly. The evaluation-set policy is also fixed (next entry
section), and every condition now reports retained n and class balance.

**Reasoning:** The review that flagged this was right that the inference was
wrong, and right about why it *could not* have been batching: class balance is
exact by construction — 640/640 dialogues in Prolific and 500/500 in SONA,
because every game contributes one human and one AI side. Verified.

But the majority detector is a **constant predictor with a slot tie-break**; it
never reads text. So nothing about filtering, silence, or batching can move it.
Only the human-in-slot-A distribution can, and that closes the case:

| half (train+dev) | human in slot A | majority slot learned |
|---|---:|---|
| prolific | 276/520 = 0.5308 | A |
| sona_ucsd | 193/391 = 0.4936 | B |

train SONA → picks B → eval Prolific = 1 − 0.5308 = **0.4692** (observed 0.4692);
train Prolific → picks A → eval SONA = **0.4936** (observed 0.4936). Both slot
splits are consistent with a coin flip (z = +1.40, −0.25). With a near-50/50
assignment, which slot counts as "majority" is itself near-random, so the
training half's majority slot is the eval half's minority slot about half the
time. Just-below-chance, no substantive cause.

**Where the review's proposed mechanism does not hold, stated because it matters
for what to fix.** The review attributed the deviation to asymmetric filtering
("the capped condition is evaluated on a subset enriched for AI"). Measured: **no
condition filtered anything.** Every ablation cell scored all 911 train+dev
games; every rung-2 direction scored 520 / 391; the counts are identical across
`A0-full`, `A0-wit-nolen` and `A0-wit-nolen-capped`. Truncating to 20 tokens does
not drop short dialogues, it leaves them short. So the capped delta was not
confounded with a drop. The review's *observation* was exactly right — 80
zero-witness dialogues, **52 human / 28 AI**, reproduced to the unit — and the
asymmetry is real; it simply was not acting through the channel proposed.

**What was wrong with the original inference, in one line:** it treated a
below-chance constant baseline as evidence about data provenance, when a constant
baseline can only report the label-position distribution. That is the same class
of error as reading a length feature as a text property — a statistic being
interpreted as something it structurally cannot measure.

**Alternatives considered:** Leaving the claim and adding a caveat — rejected;
it was an inference, not a measurement, and it was wrong. Adopting the review's
filtering explanation as given — rejected; it does not survive the counts, and
recording a second wrong mechanism to replace the first would be worse than the
original error. Dropping the majority baseline from transfer reporting — rejected;
it is informative precisely because it shows how little a constant carries.

---

### 2026-08-18 — Eval-set policy frozen (drop empty-witness games), and RUNG 4: A0 is a persona-prompt detector

**Decision:** (1) **Empty-witness games are dropped in every condition**, using
one condition-independent definition read from canonical message counts.
Retained: **851** of 911 train+dev games, exactly class-balanced (851/851; 465/465
Prolific; 386/386 SONA). Balanced accuracy is now reported alongside game
accuracy everywhere. (2) **Gate 1 rung 4 has run, and A0 does not survive a
held-out persona prompt.** Full numbers:
`v2/results/track_a/a0_rung4_20260818_writeup.md`.

**Reasoning — policy.** Silence is not a text property: a dialogue with no words
carries no lexical evidence, only a "this side went quiet" channel, and that
channel is asymmetric (of the 52 train+dev games with exactly one silent side,
the silent side was the human in 38, i.e. 73%). The drop set is defined from
canonical `n_witness_messages`, not from each condition's featurised text,
because a per-condition definition would drop fewer games in the `bothsides` cell
and silently break comparability. Measured effect of the policy: every score
moves **up**, by at most **2.2pp** — the empty games were harder than average, so
the earlier numbers were conservative rather than inflated. The silence channel's
maximum possible contribution is 38/911 = **4.17pp**, and its actual contribution
is smaller still.

**Reasoning — rung 4, and this is the finding.** Nesting first, because it is the
part that is easy to get wrong: holding out a witness system does **not** hold out
people, since the same participants play several games against different systems,
and the paired task can be won from the *human* side alone. So the component
holdout is nested inside the system holdout — train on (system ≠ held) ∧
(component ≠ held), evaluate on (system == held) ∧ (component == held). Every game
sits in exactly one cell; the six eval sets sum to 851.

Results on `A0-wit-nolen-capped`, the only condition whose signal is not length:

| holdout | TF-IDF game accuracy | balanced acc |
|---|---|---|
| people only (as reported until now) | 0.9248 [0.906, 0.941] | 0.859 |
| LOSO-system, sibling present | 0.736 – 0.902 | — |
| LOSO-system, **ELIZA** (no sibling) | **0.4259 [0.324, 0.519]**, AUROC 0.402 | — |
| **LOSO-persona** minimal→quinn | **0.6427 [0.597, 0.687]** | **0.541** |
| **LOSO-persona** quinn→minimal | **0.5799 [0.522, 0.637]** | **0.514** |

The quinn→minimal interval **contains the 0.5266 majority baseline**. Balanced
accuracy is at chance in both directions. The decisive diagnostic is side
recognition: under a held-out persona the AI side is flagged as AI in only
**24.8%** and **8.5%** of games while the human side is cleared 83.5% / 94.4% —
the model stops seeing the machine and becomes a human recogniser whose residual
edge comes from the paired comparison, not from classification. Punctuation
**inverts** (0.295 and 0.216, far below chance) in both directions, which is
direct evidence that channel encodes the prompt rather than machine-ness. And
LOSO-system tells the same story from the other side: performance tracks whether
a stylistic sibling stayed in training, and ELIZA — the only system without one —
is the only one below chance.

**Why this is the better result to have.** The 0.95 headline invited "a bag of
words beats GPTZero (89.4%) and Claude Opus 4.6 (77.9%)", a claim the corpus
cannot support: those are zero-shot detectors that never saw this data, while A0
was measured with all six witness systems in training. That comparison is now
retired. Rung 4 states what the corpus *can* support — detection of specified,
seen respondent configurations — and what it cannot. A reviewer can accept the
narrow claim; nobody should have accepted the broad one.

**Consequence.** Gate 1's stated consequence is that failed transfer is "a stop or
redesign result, not permission to fine-tune". Rungs 4–5 are non-gating by design,
so this is not an automatic stop, but **any A2 run must now be justified as
improving calibration on a within-family detector, not as building an AI
detector.** The P1 re-anchoring to calibration is better founded than when it was
made: calibration is honestly claimable on a within-family estimator, while
discrimination invites exactly the generalization claim rung 4 refutes.

**Alternatives considered:** Reporting LOSO-system only — rejected; it is the
weaker cut and would have read as a pass (0.74–0.90 excluding ELIZA), because
holding out one `quinn` system leaves the other in training. The persona cut is
what separates prompt from machine-ness, and it fails. Running rung 4 without
component nesting — rejected; the human side would leak and inflate exactly the
number under test. Treating ELIZA's below-chance score as the headline — rejected;
it is a 1966 rule-based system and its collapse is informative about sibling
dependence, not about modern-model transfer. Declaring Gate 1 failed outright —
rejected as overreach in the other direction; rungs 1–3 genuinely passed and the
within-family detector is real and well characterised. The honest statement is
narrow, not negative.

---

### 2026-08-18 — A2 collapses too. The persona ceiling is the CORPUS's, not bag-of-words'. TRACK A CONCLUDES.

**Decision:** Arm A2 ran once, as specified in §5/§13.1 — frozen temporally clean
representation, lightweight head, out-of-fold calibration — on the same two cuts
as A0 and nothing else. It **passed** the §12 feasibility check and then
**collapsed below chance** on the persona holdout. **Track A concludes.** No
further arms, no tuning, no third cut. Full numbers:
`v2/results/track_a/a2_frozen_rep_20260818_writeup.md`.

**The representation, and its temporal cleanliness.** Qwen2.5-7B, released
**2024-09-19**, already in the local Ollama store (no new download), frozen and
used only as a mean-pooling feature extractor; witness conversations were
collected **2025-03-06 … 2025-03-14**, so the checkpoint predates the data by ~6
months and cannot have memorised it. Served through a private llama-server on a
spare port so the user's running Ollama app was untouched, and stopped afterwards.

**§12 feasibility: PASS.** Train split → dev split, game accuracy **0.8319**
against a 0.5177 majority. This matters for what follows: the temporally clean
arm is *not* capability-floored, so its later collapse is a statement about the
corpus rather than about an old model failing to participate.

**The two cuts, against A0 re-run at the matched text condition** (uncapped
witness-only, so the comparison is like-for-like rather than against the
20-token-capped numbers):

| cut | A0 (TF-IDF) | A2 (frozen rep) |
|---|---|---|
| people-only | 0.9600 | **0.8660 [0.842, 0.889]** |
| LOSO-persona minimal→quinn | 0.6038 [0.558, 0.649] | **0.4929 [0.443, 0.540]** |
| LOSO-persona quinn→minimal | 0.4953 [0.437, 0.553] | **0.4483 [0.384, 0.517]** |

All four persona intervals contain 0.5; three of four point estimates are at or
below it. A2 balanced accuracy under persona holdout is 0.469 and 0.466.

**Side recognition is the decisive evidence, and it is why accuracy alone was not
allowed to settle this.** In-distribution A2 genuinely sees both sides — it flags
**74.2%** of AI dialogues as AI, clears 80.4% of humans, and separates the mean
probabilities 0.637 vs 0.378. Under a held-out persona, AI recognition falls to
**5.7%** and **6.6%**, and the two sides' mean probabilities converge to within
0.02–0.05. The estimator did not merely degrade; it became a human-recognizer
that labels almost everything human. Calibration tracks it: out-of-fold ECE
0.1008 people-only (better than A0's 0.173) versus **0.283 / 0.364** under
persona holdout — confidently wrong, the worst failure mode for the arm whose
job in this design is to own the probability.

**Answer to the question this run existed to settle.** The ceiling is the
**corpus's**. Two estimators sharing nothing but the training data — a stdlib bag
of words and a frozen 7B neural representation that demonstrably recognises AI
in-distribution and is temporally incapable of contamination — fail the same way
on the same cut, and the neural one fails *harder*. That locates the limit in
what the data can teach, not in how the text is represented. The most likely
reading is that a five-minute adversarial transcript carries far more signal
about which prompt the witness was given than about whether the witness is a
machine.

**GATE 1 VERDICT.** This corpus supports the study of passive identity judgment
against *specified, seen* respondent configurations, and supports calibration
work on that task: the signal is real (A0 0.960, A2 0.866 with ECE 0.101 and
two-sided recognition), and it is not an artefact of interrogator text, reply
rate or length, redaction placeholders, or respondent silence — each measured and
excluded — and it transfers between recruitment populations. It does **not**
support any claim about detecting AI respondents in general, about unseen model
or prompt configurations, or about deployment against an adversary free to choose
its own prompt. Gate 1's stated consequence for failed transfer is "a stop or
redesign result, not permission to fine-tune", and that is the outcome on the
axis that matters most for v2.1's question.

**What this does not forbid.** An A2-style estimator remains usable as the frozen
probability authority for *within-configuration* calibration studies — exactly
the role §5 assigns it, and the reason the earlier P1 re-anchoring from
discrimination to calibration now reads as the right call. What is refuted is any
use of it as an AI detector.

**Alternatives considered:** Reporting A2's people-only 0.866 as a weaker result
than A0's 0.960 and concluding the representation was inadequate — rejected; the
feasibility check passed at 0.832 and in-distribution side recognition is 74.2%,
so A2 demonstrably has the capability whose disappearance is being measured.
Attributing A2's collapse to the 512-d random projection — rejected; the same
projection is present in the people-only cut where A2 scores 0.866, so it cannot
manufacture a persona-specific collapse, and it can only lose information, making
it conservative. Adding a dedicated embedding model — rejected; it would need a
download, and the instruction was one run, no new arms. Trying a stronger head, a
different regulariser, or a third cut — rejected outright; that is tuning against
a result, and the session was specified as one run precisely to prevent it.
Concluding that Track A produced nothing — rejected as overreach in the other
direction: a characterised within-family detector with measured calibration and a
precisely located generalization boundary is a usable result, and the boundary is
more valuable than the accuracy number would have been.

---

### 2026-08-18 — CORRECTION: "the ceiling is the corpus's" narrowed to a two-method result

**Decision:** The claim in the previous entry — *"the ceiling is the **corpus's**"*,
including the entry's own title — is **narrowed**. The supported statement is:
*two methods, a TF-IDF baseline and one frozen Qwen2.5-7B representation with a
logistic head, both fail to transfer between these two prompt families, and the
neural one fails harder.* Correction sheet:
`v2/results/track_a/corrections_20260818_213000.md` (§C1). The code docstring in
`track_a_a2.py` is corrected in place; the append-only writeups are corrected by
the sheet, not edited.

**Reasoning:** "the corpus's ceiling" is an impossibility claim about all methods,
resting on n = 2. The second method carries three handicaps a stronger A2 would
not: crude mean-pooling as a sentence representation, a 512-d random projection
adopted purely because this venv has no BLAS, and a single untuned head at one
regularisation setting. A1 and A3 were never run. A different representation, a
nested calibrator, or turn-wise evidence could in principle recover signal this
pair missed.

What survives is the part that actually justified stopping: the failure is **not
an artefact of lexical representation**, because swapping the representation
entirely did not help — it hurt. That is strong evidence of severe prompt
dependence, and it is enough to stop Track A. It is not proof that no learnable
general signal exists here, and the difference between those two statements is
exactly the kind of overreach this project has twice had to retract.

**Alternatives considered:** Editing the A2 writeup in place — rejected;
`v2/results/` is append-only and the sequence claim → correction is itself the
useful record. Defending the original on the grounds that two very different
methods is suggestive — rejected; suggestive is not what "the corpus's ceiling"
asserts, and the honest fix is to say the weaker thing. Re-running A2 with a
better representation to settle it — rejected here; that is new experimental work
on a concluded track, and the reconciliation must land before any further runs.

---

### 2026-08-18 — CORRECTION: A2 is a probabilistic head evaluated for calibration, not a calibrated head

**Decision:** A2 is renamed from "calibrated classifier head" to **"probabilistic
head, evaluated for calibration"** in `track_a_a2.py` (docstring and the `head`
field of its output JSON) and corrected for the writeups by
`corrections_20260818_213000.md` (§C2). **No calibrator is added.** All reported
Brier/ECE/reliability numbers stand unchanged.

**Reasoning:** `track_a_a2.py` fits an ordinary L2 logistic regression and
*measures* calibration on out-of-fold predictions. It fits no Platt, isotonic, or
temperature stage nested inside the training folds. **Out-of-fold calibration
evaluation is not out-of-fold calibration**, and design §13.1 asks for the latter
("calibrated full-history classifier head"). So the arm as built is a deviation
from the plan, and describing it as calibrated claimed a component that does not
exist.

The measurements are unaffected: people-only dialogue Brier 0.1663 / ECE 0.1008,
persona holdout ECE 0.283 and 0.364. They are legitimate statements about how
well this head's raw probabilities happen to be calibrated. The A0-vs-A2 ECE
comparison (0.173 vs 0.101) also stands, because both are uncalibrated estimators
measured identically.

**Why no calibrator was built.** The review asked for the description to be
corrected first, and fitting one now would be new experimental work on a track
that has stopped. It is recorded as outstanding, not done. A properly nested
calibrator would fit on the training folds only, never on the held-out fold, or
it would leak the very quantity it is meant to estimate.

**Alternatives considered:** Adding a Platt stage immediately so the original word
becomes true — rejected; that is fixing the evidence to match the claim rather
than the claim to match the evidence, and it would restart experiments before the
record is reconciled. Dropping calibration reporting entirely — rejected; the
measurements are informative, and the persona-holdout ECE of 0.283/0.364
(confidently wrong) is one of the sharper results in the session.

---

### 2026-08-18 — PLAN REVISION: Track A concluded early; A1 skipped, A3 deferred; the non-gating contradiction resolved

**Decision:** Track A is recorded as **concluded early relative to the frozen
plan**, not complete. A1 is **SKIPPED**, A3 is **DEFERRED** to optional Stage D.
Gate 1 is restated as: **broad Gate 1 failed or unavailable; within-configuration
passive detection established.** This is a deviation from `v2/experiment_design.md`
§13.1 and §16 and is labelled as one. Detail: `corrections_20260818_213000.md`
§§C3–C4.

**Reasoning — the bookkeeping.** §13.1 lists four Track A arms. A0 ran. A2 ran
(uncalibrated, see the previous entry). **A1 and A3 were never implemented.** A1's
non-implementation was declared up front and for good reasons — its 557 games are
48.9% of the corpus, biased long, and the A2-vs-A1 contrast had already been
demoted to interval reporting — but a well-justified skip is still a skip, and
"Stage B complete" was wrong while two planned arms were outstanding.

**Reasoning — Gate 1, and why "failed" alone would be misleading.** Gate 1 asks
for "useful out-of-source discrimination and calibration". **No out-of-source
holdout was ever available**: the only other real corpus in hand is the 15-minute
study, whose Gate 0 is unresolved, so it was never read. Every holdout actually
run was within-corpus. So Gate 1 is not passed *partly by failure* (the persona
rung) and *partly by unavailability* (no second source), and collapsing those two
into one verdict would misdescribe what the project learned.

**Reasoning — the contradiction, which is the substantive part.** The rung-4 entry
records persona and family rungs as **non-gating** ("reported but not gating").
The next entry then used persona failure to declare Track A concluded. Both cannot
stand as written. **Resolution: the stop was a judgement call informed by a
non-gating rung, not a gate firing.** Persona failure did not automatically fail
Gate 1. It made further Track A investment unattractive, because the axis that
failed is precisely the one v2.1's question depends on — an investigator that
cannot generalize past the respondent's prompt cannot support the calibrated
active-detection claim the project exists to test. Stopping on that basis is
defensible; presenting it as the plan executing as written is not. Hence a
recorded revision.

**What this does not change.** No measurement, and not the stop itself. Track A
still stops, and the reconciliation queue in `STATUS.md` still gates any further
experiment. What changes is that a later reader can see which arms ran, which did
not, which gate conditions were untestable, and that the stopping rule applied was
judgement rather than a threshold.

**Alternatives considered:** Running A1 and A3 now to make "complete" true —
rejected; both were declined on their merits, the track has stopped, and running
arms to satisfy a word is code-first reasoning. Declaring Gate 1 simply "failed" —
rejected; it understates, because the required out-of-source test was never
available to fail. Declaring Gate 1 "passed on rungs 1–3" — rejected; it
overstates, because rung 3's own wording makes artefact-survival necessary rather
than sufficient, and the persona result is the decision-relevant one. Leaving the
contradiction unresolved with a footnote — rejected; an unresolved contradiction
between two adjacent log entries is exactly what makes a log stop being trusted.
