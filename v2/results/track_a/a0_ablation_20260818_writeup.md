# Track A, arm A0 — three-way ablation, and Gate 1 rung 2

**Date:** 2026-08-18 · **Stage:** B · **Supersedes nothing; appends to**
`a0_baselines_20260818_writeup.md`, which is left unedited (results/ is
append-only).

**Runs:** `a0_baselines_20260818_155706_ablation.json` (4 cells),
`a0_baselines_20260818_160542_capped.json` (length-equalised control),
`a0_rung2_transfer_20260818_160816.json` (rung 2, both directions).

```
canonical source_revision 6891d9a8a2c4   split main_study_v1 256543688a5da358…
```

**Test split untouched.** No GPU, no inference, no downloads.

---

## 0. Headline

**The signal survives both ablations. It is genuinely lexical, not an artefact of
interrogator text or of reply rate.**

- Artefact (a) — interrogator turns in the classified text — **was never present
  in this pipeline**. Verified, not assumed: the loader builds from
  `tt_message_anonymized.sender_role`, never from the `transcript` column.
  `A0-full` and `A0-witness` are byte-identical on every metric.
- Artefact (b) — length as reply rate — **is real and is confirmed in full**,
  including the decisive detail. It destroys the `length` family (0.827 → 0.500,
  every feature removed) and costs the others 0.7–2.0pp.
- With every length feature removed **and** every witness side truncated to a
  fixed 20-token budget, TF-IDF still scores **0.9210 [0.9032, 0.9383]**.
- **Gate 1 rung 1 stands**, but its basis changes: it now rests on the
  length-free number, not on the un-ablated headline. The `length` row of the
  earlier writeup should not be cited as a detector result.
- Pre-registered floor (0.65, CI lower bound, written before the run) **cleared**
  → rung 2 ran. Transfer holds: **0.9250 / 0.9540** in the two directions.

---

## 1. Premise verification

Both claims were checked against the source before any of them was relied on.

### (a) The transcript column interleaves both sides — TRUE, and irrelevant here

```
first tt_transcripts row, role counts : Counter({'I': 17, 'W': 17})
across all 2,282 transcript rows      : {'I': 9277, 'W': 8811}
```

Exactly as stated. **But this pipeline never reads that column.**
`build_canonical.py` builds `turns` from `tt_message_anonymized.csv` using the
`sender_role` field, and `dialogue_messages()` filters to `role == "W"`.

Proved rather than asserted — every dialogue's featurised text was compared
against witness-only content reconstructed independently from the source message
table:

```
dialogues checked                      : 2280
featurised text != witness-only source : 0
source witness msgs 8796 / interrogator msgs 9262 / featurised msgs 8793
```

(The 3-message gap is whitespace-only bodies that collapse to empty; the 6
empty-content rows in the release.) So the concern was well-founded about the
released artefact and **did not apply to these numbers**. `A0-full` ≡ `A0-witness`
is the empirical confirmation, and it is reported as a cell rather than skipped.

### (b) Length encodes reply rate — TRUE, every figure reproduced

| quantity | human conv. | AI conv. |
|---|---:|---:|
| median **witness** words | 24 | 31 |
| median **interrogator** words | **20** | **26** |
| median witness messages | 3 | 4 |
| median total messages / conversation | 6 | 8 |

`tt_aimodel`: `max_length` = 1000 for **all 12** configs; `secs_per_char` = 0.3
for **all 12**. So it is not a generation cap.

**The interrogator row is the proof, and it is worth stating plainly.** The same
person runs both conversations of a game, simultaneously, under one 5-minute
clock. Nothing about *them* differs between the two. Yet they write 20 words
opposite a human and 26 opposite an AI. A difference in the interrogator's own
output can only be caused by how fast the other side replied. Length here is a
property of the apparatus — one no deployed detector would have.

---

## 2. TASK 1 — the ablation table

911 games (train+dev), 11-fold leave-one-component-out, raw text variant, 1,000
bootstrap replicates, seed 20260817. **Independence unit: the participant**,
reported as the wider of the interrogator-clustered and human-witness-clustered
marginal bootstraps (games are crossed on both roles; a true two-way crossed
bootstrap is not implemented). Cells are game accuracy with 95% CI.

| condition | length | punctuation | function words | TF-IDF LR |
|---|---|---|---|---|
| **A0-full** (as run) | 0.8266 [0.800, 0.852] | 0.7903 [0.760, 0.822] | 0.8463 [0.823, 0.868] | 0.9539 [0.939, 0.967] |
| **A0-witness** | 0.8266 [0.800, 0.852] | 0.7903 [0.760, 0.822] | 0.8463 [0.823, 0.868] | 0.9539 [0.939, 0.967] |
| **A0-wit-nolen** | **0.5000** † | 0.7827 [0.751, 0.814] | 0.8260 [0.803, 0.849] | 0.9539 ‡ [0.939, 0.967] |
| *A0-wit-nolen-capped* § | 0.5000 † | 0.7124 [0.682, 0.743] | 0.8019 [0.772, 0.830] | **0.9210 [0.903, 0.938]** |
| *A0-bothsides* ¶ | 0.7728 [0.745, 0.802] | 0.7080 [0.676, 0.739] | 0.8052 [0.780, 0.831] | 0.9484 [0.934, 0.962] |

Baselines, all conditions: random 0.5000, majority 0.5148.

† **No features remain.** All ten length features are length-derived, so the
family is empty under `nolen` and the detector degenerates to the class prior —
constant output, every game a tie, 0.5 credit each. This is not a measured score;
it is the arithmetic consequence of the family being 100% length, which is
exactly what artefact (b) predicts.

‡ **`drop_length` is a NO-OP for TF-IDF, and the identical 0.9539 must not be
read as "TF-IDF survived length removal".** TF-IDF has no explicit length
features to remove. Its vectors were already L2-normalised — verified
empirically, all document norms exactly 1.000000000 — so the vector-norm channel
the task flagged was already closed. But a longer document still activates *more
distinct terms* after L2 normalisation, and `drop_length` cannot reach that. Row
§ exists because of this.

§ **Not requested; added because without it the TF-IDF row above is
uninterpretable.** Every witness side is truncated to a fixed 20-token budget,
which equalises length directly: median witness words go from 24 (human) / 30
(AI) to **20 / 20**. The budget was picked from the measured medians (just under
the human median so most dialogues on both sides actually reach it), not tuned
against any score. Residual: dialogues shorter than 20 tokens are not padded, so
means remain 15.74 / 17.30 rather than equal. **The drop from 0.9539 to 0.9210 is
an upper bound on the length channel**, because truncation removes information as
well as equalising length.

¶ Diagnostic: what a loader reading the `transcript` column would have scored.
Including interrogator turns **hurts** every family (TF-IDF AUROC 0.9515 →
0.9261; punctuation 0.7776 → 0.6654). Artefact (a) would have *diluted* the
signal, not inflated it — consistent with the weak direct label-leak measured in
`DECISIONS.md` (bot/AI mentions 14.3% vs 16.3% of conversations).

Features dropped under `nolen`, recorded so the ablation is auditable:

```
length         10/10 dropped: empty frac_long frac_short max_chars mean_chars
                              mean_words n_messages sd_chars total_chars total_words
punctuation     1/19 dropped: empty
function_words  2/116 dropped: empty fw_type_token
```

`fw_type_token` (type-token ratio) is dropped as a length proxy: TTR falls with
text length by construction, so it is a length feature wearing a
lexical-diversity costume.

---

## 3. TASK 2 — interpretation

### The signal is genuinely lexical/semantic. Said plainly.

Three independent feature families that contain **no length information at all**
retain strong discrimination, and two of them are pure rates:

- **function words** — rates per 1,000 tokens, no TTR, no message count:
  **0.8260**, and **0.8019** even on 20-token snippets.
- **punctuation** — rates per 1,000 characters and per-message fractions:
  **0.7827**, falling to 0.7124 when capped.
- **TF-IDF** on length-equalised 20-token snippets: **0.9210 [0.9032, 0.9383]**,
  AUROC 0.9217.

A model that sees twenty words from each side, with all length cues removed,
picks the human out of the pair 92% of the time. That is a lexical/semantic
signal, not an apparatus artefact.

### This beats published numbers on related data, and that is a claim requiring rung 4 FIRST

Inverse Turing Bench, on dialogue from this same corpus, reports **GPTZero 89.4%**
and **Claude Opus 4.6 77.9%**. A stdlib bag-of-words logistic regression at
0.954 — or 0.921 with every length cue stripped — beats both.

The tasks are not identical (ITB is paired discrimination over its 557-game
long-biased subset; A0 here is scored on all 911 train+dev games), so the
ordering is not impossible. But per the task framing and the 2026-08-07
precedent, **this makes it a claim that needs the witness-system decomposition
(rung 4) before publication, not after.** The reason is visible in the data
already:

| witness system | TF-IDF, in-source | SONA→Prolific | Prolific→SONA |
|---|---:|---:|---:|
| eliza | 0.987 | 0.992 | 0.981 |
| gpt-4.5_quinn (persona) | 0.957 | 0.948 | 0.958 |
| llama-405b_quinn (persona) | 0.964 | 0.961 | 0.958 |
| gpt-4.5_minimal | 0.960 | 0.903 | 0.962 |
| gpt-4o_minimal | 0.924 | 0.856 | 0.957 |
| llama-405b_minimal | 0.915 | **0.828** | 0.896 |

Folds hold out **people, not systems**: every witness system appears in every
training fold. So A0 is a six-system classifier, and the aggregate is a weighted
average over six systems the model has seen. The no-persona systems are already
the weakest and degrade most under transfer (llama-405b_minimal 0.915 → 0.828).
A seventh, unseen system is untested. Until rung 4 runs, the honest description
is *"distinguishes these six witnesses from these humans"*, not *"detects AI"*.

### Verdict on Gate 1 rung 1

**Rung 1 stands, on a corrected basis.** The earlier writeup's "PASSES" line is
not retracted — the ablation did not collapse — but two corrections attach:

1. The `length` family's 0.827 is **not a detector result**. It is a measurement
   of reply-rate asymmetry and should never be cited as evidence of detection.
2. The number rung 1 rests on should be the length-free one (0.9539 with the
   no-op caveat, or 0.9210 under the strict capped control), not the un-ablated
   0.954.

---

## 4. Re-checking the P1 ceiling argument against the ablated A0

`DECISIONS.md` (2026-08-18, "P1 re-anchored") argued that P1's frozen 10pp
accuracy target was unpassable because A0 sat at 95.4%, leaving ≤4.6pp of
headroom, and required that argument be re-checked after the ablation.

**Re-checked. The ceiling argument survives, and it does not depend on which
ablated number is chosen:**

| A0 variant used as the baseline | A0 | headroom to 1.0 |
|---|---:|---:|
| A0-full (as run) | 0.9539 | 4.6pp |
| A0-wit-nolen | 0.9539 | 4.6pp |
| A0-wit-nolen-capped (strictest) | 0.9210 | 7.9pp |

Even under the strictest control the headroom is **7.9pp against a 10pp target** —
still structurally unpassable. The re-anchoring from discrimination to
calibration therefore stands and needs no revisiting. **This is the "once, in
writing, before any A2 run" check that entry required, and it is now discharged.**

Calibration remains wide open, which is the point: TF-IDF ECE is **0.173**
(un-ablated and nolen) and **0.141** capped, while function words sit at
0.045–0.063. The best discriminator is still the worst-calibrated estimator.

---

## 5. TASK 3 — Gate 1 rung 2, SONA/UCSD ↔ Prolific

The pre-registered floor (recorded in `DECISIONS.md` **before** the run: best
detector's participant-clustered CI lower bound > 0.65) was cleared —
A0-wit-nolen TF-IDF lower bound **0.9394**, and the best *genuinely ablated*
family, function words, **0.8035**. Cleared under either reading, so rung 2 ran.

Preconditions verified independently before running:

- **All six witness systems present in both sources** (the task said four; it is
  six), Prolific share 0.54–0.59 per system — so a transfer drop is not explained
  by one population meeting different AI systems.
- **Recruitment source is perfectly nested in components: 0 of 15 mixed.**

Single fit per direction — no cross-fitting needed, because the two populations
are participant-disjoint by construction.

| condition | direction | n train / eval | function words | TF-IDF LR |
|---|---|---|---|---|
| A0-full | SONA→Prolific | 391 / 520 | 0.7990 [0.767, 0.829] | 0.9250 [0.901, 0.948] |
| A0-full | Prolific→SONA | 520 / 391 | 0.8120 [0.772, 0.850] | 0.9540 [0.932, 0.974] |
| **A0-wit-nolen** | SONA→Prolific | 391 / 520 | 0.7808 [0.748, 0.813] | **0.9250 [0.901, 0.948]** |
| **A0-wit-nolen** | Prolific→SONA | 520 / 391 | 0.8018 [0.764, 0.838] | **0.9540 [0.932, 0.974]** |
| A0-wit-nolen-capped | SONA→Prolific | 391 / 520 | 0.7731 [0.736, 0.814] | 0.8750 [0.847, 0.903] |
| A0-wit-nolen-capped | Prolific→SONA | 520 / 391 | 0.8018 [0.760, 0.841] | 0.9207 [0.891, 0.946] |

**Transfer holds.** Against the in-source cross-fitted A0-wit-nolen figure of
0.9539, transfer costs 2.9pp in one direction and nothing in the other. Under the
strict capped control, 3.4pp/0.0pp against its own 0.9210 in-source figure.

**How this must be described.** It is a **between-experiment holdout, confounded
with batch/lobby structure**. Because recruitment source is perfectly nested
inside the components, "different people" and "different collection batch" move
together and nothing in the corpus separates them. It is **not** a clean
population holdout, and it is **not** a dataset/source holdout — both halves
share one apparatus, one time window, and one witness-system set, so it cannot
detect a collection artefact common to both. That remains rung 3's job.

One incidental result worth recording: the **majority baseline goes below chance
in transfer** (0.4692 and 0.4936, from 0.5148 in-source). The majority human slot
position learned from one population does not hold in the other, which is a small
independent confirmation that the two halves really are separately batched.

---

## 6. What this session establishes, and what it does not

**Establishes:**

- The interrogator-contamination artefact was absent from this pipeline (proved).
- The reply-rate artefact is real, fully accounts for the `length` family, and
  costs the other families 0.7–2.0pp.
- A length-free, witness-only lexical signal of ~0.92–0.95 exists and transfers
  across the two recruitment populations.
- Gate 1 rungs 1, 2 and 3 now all have evidence. The P1 ceiling argument survives.

**Does not establish:**

- **Anything about an unseen witness system.** Rung 4 is not run. Every number
  here is in-distribution with respect to the six systems.
- **That the detector would work on ordinary conversation.** The human class is
  people under adversarial interrogation on a 5-minute clock (design §4).
- **That the 20-token capped number isolates the length channel exactly.** It is
  an upper bound on that channel; truncation also removes information.
- **A dataset-level holdout.** Needs the 15-minute study, whose Gate 0 is
  unresolved.
- **Calibration.** ECE 0.14–0.17 for the best discriminator, out of fold, and no
  calibration layer has been fitted.

---

## 7. Reproducing

```bash
venv/bin/python v2/scripts/build_canonical.py
venv/bin/python v2/scripts/track_a_a0.py --ablation --bootstrap 1000 --tag ablation
venv/bin/python v2/scripts/track_a_a0.py --conditions A0-wit-nolen-capped \
    --variants raw --bootstrap 1000 --tag capped
venv/bin/python v2/scripts/track_a_rung2.py --bootstrap 1000
venv/bin/python -m unittest tests.test_v2_track_a          # 47 tests
```

`track_a_a0.py` with no arguments still reproduces the original
`a0_baselines_20260818_010835.json` behaviour: the defaults are witness-only, no
length dropping, no token cap. ~9 minutes total, stdlib only.
