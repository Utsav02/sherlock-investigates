# Stage A data inspection — Jones & Bergen three-party Turing test (OSF `jk7bw`)

**Date:** 2026-08-17 · **Stage:** A, steps 3–4 of `v2/experiment_design.md` §18
**Source:** `v2/data/sources/jones_bergen_2025/` (29 files, 19,660,453 bytes,
downloaded 2026-08-17T21:16:26Z; every sha256 matches the OSF API — see
`MANIFEST.json`). Licence/consent record:
`v2/data/sources/registry/jones_bergen_2025.md`.

**Every number below is measured from the downloaded files**, by
`v2/scripts/inspect_three_party.py` (`make v2-inspect-3p`), and is reproducible
from `three_party_inspection_data.json` / `three_party_inspection_15_mins.json`
in this directory. Numbers from the papers appear only where they are explicitly
labelled as such, and only to be contrasted with what the files contain.

> **Headline for planning:** the release is *two* studies, not one. The main
> 2025 study (`data/`, 1,140 games, median **8 messages per conversation**) and
> a previously unmentioned 15-minute study (`15_mins/`, 392 games, median **22
> messages per conversation**, with a **GPT-5** witness, collected January 2026).
> Track A gets ~3× more conversational evidence per game from the second one.

---

## 1. File inventory and schema

### 1.1 `data/` — main study (5-minute games, March 2025)

| File | Bytes | Rows | Columns |
|---|---:|---:|---|
| `tt_game.csv` | 146,384 | 1,140 | id, interrogator_id, human_witness_id, ai_witness_id, status, created_at, finished_at, verdict_at, last_updated, is_test |
| `tt_conversation.csv` | 55,106 | 2,280 | id, game_id, interrogator_id, witness_id, label |
| `tt_witness.csv` | 9,599,585 | 2,280 | id, witness_type, user_id, lobby_wait_id, ai_model_id, prompt, name, city, country, languages, timezone |
| `tt_interrogator.csv` | 18,271 | 1,140 | id, user_id, lobby_wait_id |
| `tt_message_anonymized.csv` | 1,621,207 | 18,058 | id, conversation_id, sender_role, timestamp, cost, hidden, content, is_changed |
| `tt_verdict.csv` | 139,301 | 1,141 | id, game_id, interrogator_id, verdict, confidence, reason, is_correct, created_at |
| `tt_transcripts.csv` | 990,362 | 2,282 | game_id, conversation_label, is_human, witness, message_count, verdict, confidence, reason, is_correct, transcript |
| `tt_profile.csv` | 100,817 | 323 | id, user_id, gender, year_of_birth, education, chatbot_interaction_frequency, familiarity_with_GPT, expt_aware, expt_aware_details, source, study, sona_credit_granted, emotion, intelligence, strategy, strategy_change, accuracy_estimate, other |
| `tt_aimodel.csv` | 1,899 | 12 | id, name, family, model_name, active, prompt_id, prompt_extra, temperature, secs_per_char, max_length, response_method, wins, losses, weighted_wins, weighted_losses, batch, city |
| `tt_prompt.csv` | 10,931 | 5 | id, label, family, version, prompt, created, last_updated, notes |
| `tt_games_enriched.csv` | 752,564 | 1,140 | 44 columns — a pre-joined convenience table (game × verdict × interrogator profile × model) |
| `tt_strategyclass.csv` | 4,663 | 26 | id, name, description, category, code, is_active, created, updated |
| `3p_reason_scheme.csv` | 2,592 | 23 | class, category, short_name, description |
| `reason_classifications_4o.csv` | 22,524 | 1,500 | game_id, class, tagged_by |
| `strategy_classifications_4o.csv` | 31,374 | 2,090 | game_id, class, tagged_by |
| `codebook.txt` | 15,642 | — | table-by-table variable documentation |

Plus, at the node root: `codebook.md` (AI-model table only) and three analysis
notebooks — `3p_sona_preregistered_analyses_clean.Rmd`,
`3p_prolific_preregistered_analyses_clean.Rmd`, `3p_combined_plots_clean.Rmd`.

Game timestamps span **2025-03-06T17:55:47Z → 2025-03-14T16:29:20Z**. All 1,140
rows have `status = verdict` and `is_test = FALSE`.

### 1.2 `15_mins/` — 15-minute study (January 2026), a separate study

| File | Bytes | Rows |
|---|---:|---:|
| `tt_game.csv` | 50,288 | 392 |
| `tt_conversation.csv` | 23,564 | 784 |
| `tt_witness.csv` | 2,920,529 | 784 |
| `tt_interrogator.csv` | 9,041 | 392 |
| `tt_message_anonymized.csv` | 1,897,980 | 18,933 |
| `tt_transcripts.csv` | 1,003,285 | 784 |
| `tt_profile.csv` | 69,214 | 227 |
| `tt_aimodel.csv` | 924 | 5 |
| `codebook.txt` | 12,204 | — |

Same schema as the main study **except there is no `tt_verdict.csv`** — its own
codebook documents a verdict table, but the file is absent from the release, so
the verdict, confidence and reason survive only on the transcript rows
(duplicated across the game's A and B rows). Game timestamps span
**2026-01-13T18:46:59Z → 2026-01-14T14:50:35Z**. Its codebook states the export
already applied the analysis exclusions (completed games, date window,
duration < 1000 s, Prolific only, experiment-aware participants dropped, games
with a conversation under 2 messages dropped).

### 1.3 Schema gotcha that silently breaks joins

In `15_mins/`, ids are written through pandas floats: `tt_witness.id`,
`tt_witness.user_id`, `tt_witness.ai_model_id` and the witness keys in
`tt_game.csv` appear as `109819.0` where `tt_profile.user_id` has `109819`.
Measured count of affected key values: **784 in `tt_witness`, 392 in `tt_game`**;
zero in the main study. A naive string join therefore reports **0 human-witness
participants and 0 games per AI model** in the 15-minute study — which is what
this inspection produced on its first pass. `inspect_three_party.norm_id`
normalises them; any canonical loader must do the same.

---

## 2. Participants and identifiers

| | main study | 15-minute study |
|---|---:|---:|
| `tt_profile` rows / distinct `user_id` | 323 / 323 | 227 / 227 |
| Distinct users appearing as interrogator | 306 | 204 |
| Distinct users appearing as human witness | 314 | 215 |
| Distinct users in either role | **323** | **227** |
| Users appearing in both roles | 297 | 192 |
| Recruitment source (`source`) | 183 Prolific (1), 140 SONA/UCSD (2) | 227 Prolific (1) |

**Participants have stable identifiers, but not the obvious ones.**
`tt_interrogator.id` and `tt_witness.id` are per-game *seat* ids (1,140 and
2,280 rows for 323 people). The person-level key is **`user_id`**, which links
`tt_profile` ↔ `tt_interrogator` ↔ human `tt_witness`. Human-witness rows carry
`user_id`; AI-witness rows have `ai_model_id` instead. This is sufficient for
the §14 rule that one participant belongs to one split — provided the join uses
`user_id` and the float normalisation above.

Demographics are present for 307/323 profiles in the main study (gender, year of
birth, education, chatbot-use frequency, LLM familiarity) and 209/227 in the
15-minute study; `emotion`, `intelligence` and `accuracy_estimate` are present
for all rows.

### Games per participant (a clustered design, by construction)

Main study:

| Role | n users | mean | sd | min | median | max |
|---|---:|---:|---:|---:|---:|---:|
| as interrogator | 306 | 3.73 | 0.69 | 1 | 4 | 4 |
| as human witness | 314 | 3.63 | 0.81 | 1 | 4 | 4 |
| either role | 323 | 7.06 | 1.98 | 1 | 8 | 8 |

Interrogator-game histogram: 254 users with 4 games, 30 with 3, 12 with 2, 10
with 1. 15-minute study: 1.92 games per interrogator (max 2), 3.45 per user
across both roles.

**Implication for Track A precision (design §15):** with ~4 judged games per
interrogator and the same people appearing as witnesses, rows are clustered
twice over — by interrogator and by human witness. Any interval must be
clustered; the effective n for interrogator-level contrasts is ~306, not 1,140.

---

## 3. Games

| | main study | 15-minute study |
|---|---:|---:|
| Rows in `tt_game.csv` | **1,140** | **392** |
| Distinct game ids in `tt_transcripts` | 1,140 | 392 |
| Transcript rows | 2,282 (2 duplicates, §7) | 784 |
| Verdict rows | 1,141 (1 duplicate, §7) | n/a (in transcripts) |

**Contrast with the papers, stated as a contrast only:** arXiv:2503.23674
reports 1,023 analysed games from 284 participants (445 games / 126 UCSD
participants; 576 games / 158 Prolific participants). The release holds 1,140
completed games and 323 profiles, i.e. it is the *pre-exclusion* export. Applying
the stated exclusions naively does not reproduce 1,023: excluding games whose
interrogator is flagged `expt_aware` gives 1,052, and additionally requiring ≥2
messages in each conversation gives **985**; excluding on either participant's
awareness gives 976 / 911; requiring ≥1 interrogator message per conversation
gives 1,014. **Stage B must reproduce the analysed subset by running the
released `.Rmd` scripts, not by re-deriving the filter.**

---

## 4. Witness systems

Main study — 12 `tt_aimodel` rows, which are 6 systems × 2 recruitment
populations (SONA/Prolific prompt variants):

| Witness (transcript label) | Model id(s) | `model_name` | Games | Interrogator correct |
|---|---|---|---:|---:|
| `gpt-4.5_quinn` (PERSONA) | 73, 84 | gpt-4.5-preview-2025-02-27 | 286 | 78 (**0.273**) |
| `llama-405b_quinn` (PERSONA) | 76, 82 | Meta-Llama-3.1-405B-Instruct-Turbo | 275 | 121 (0.440) |
| `gpt-4.5_minimal` (NO-PERSONA) | 71, 85 | gpt-4.5-preview-2025-02-27 | 154 | 98 (0.636) |
| `eliza` | 80, 81 | eliza | 147 | 112 (0.762) |
| `gpt-4o_minimal` | 50, 86 | gpt-4o-2024-08-06 | 141 | 111 (0.787) |
| `llama-405b_minimal` | 74, 83 | Meta-Llama-3.1-405B-Instruct-Turbo | 137 | 84 (0.613) |

(The measured 0.273 correct-identification rate against GPT-4.5-PERSONA is the
complement of the paper's headline 73% win rate — a useful sanity check that the
join is right.)

15-minute study — 5 model rows, 3 systems:

| Witness | `model_name` | Games | Interrogator correct |
|---|---|---:|---:|
| `llama-405b_quinn` | Meta-Llama-3.1-405B-Instruct-Turbo | 149 | 66 (0.443) |
| `gpt-5-quinn` | **gpt-5-chat-latest** | 148 | 61 (0.412) |
| `eliza` | eliza | 95 | 79 (0.832) |

**Family support for holdout (design §14):** the main study has two substantive
families (OpenAI: 581 games across GPT-4.5 and GPT-4o; Meta-Llama: 412) plus
ELIZA (147) as a weak control. That is two families. Leave-one-family-out is a
two-point comparison and must be reported as exploratory. Adding the 15-minute
study brings GPT-5 in, but as a different study with a different time limit —
a confound, not a free third family.

Every AI witness row carries its full system `prompt` (1,140 non-empty in the
main study, 392 in the 15-minute study), and `tt_prompt.csv` holds the 5 prompt
templates. Persona/no-persona is therefore recoverable per game — needed for the
persona-degradation analysis in §15 of the design.

---

## 5. Messages, turns, and message length

| | main study | 15-minute study |
|---|---:|---:|
| Message rows | 18,058 | 18,933 |
| Interrogator (`I`) / witness (`W`) | 9,262 / 8,796 | 9,591 / 9,342 |
| Conversations with ≥1 message | 2,232 of 2,280 | 784 of 784 |
| `hidden = TRUE` | 0 | 0 |
| Empty `content` | 6 | 1 |

**Interrogator and witness turns are unambiguously distinguishable**, two ways:
the `sender_role` column (`I`/`W`) in the message table, and the `I: ` / `W: `
line prefixes in the joined `tt_transcripts.transcript` string. Both agree.
Which conversation held the human is given by `tt_transcripts.is_human` and by
`tt_witness.witness_type` (`H`/`A`); message timestamps are present on every row,
so turn order and latency are recoverable.

Messages per **conversation** (one witness):

| | n | mean | sd | min | p25 | median | p75 | p90 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| main, all | 2,232 | 8.09 | 4.59 | 1 | 5 | **8** | 10 | 14 | 39 |
| main, AI witness | 1,115 | 8.96 | 5.01 | 1 | 6 | 8 | 12 | 16 | 39 |
| main, human witness | 1,117 | 7.22 | 3.93 | 1 | 4 | 6 | 10 | 12 | 34 |
| 15-min, AI witness | 392 | 29.66 | 11.82 | 3 | 21 | 28 | 38 | 46 | 64 |
| 15-min, human witness | 392 | 18.64 | 7.58 | 4 | 14 | 18 | 23 | 29 | 51 |

Messages per **game** (both conversations): main study mean 15.95, median 15,
range 2–56 over the 1,132 games that have messages; 15-minute study mean 48.30,
median 47, range 7–100.

Message length, main study:

| Role | n | mean chars | median chars | max chars | mean words | median words | max words |
|---|---:|---:|---:|---:|---:|---:|---:|
| Interrogator | 9,262 | 31.5 | 26 | 282 | 6.4 | 5 | 55 |
| Witness | 8,796 | 41.0 | 29 | 303 | 8.0 | 6 | 60 |

15-minute study: interrogator 42.1 chars / 8.5 words mean, witness 44.5 chars /
8.9 words mean, witness max 676 chars. Both codebooks state "Messages were
limited to 300 characters"; measured, the main study has 2 messages over 300
(max 303) but the 15-minute study has **21** (17 witness, 4 interrogator; max
676), so the documented cap does not describe that study's data. A length
feature is therefore a study-level artefact as well as a writer property.

Whole-transcript size (one side of a game), main study: median **61 whitespace
tokens**, mean 65.0, max 209; median 296 characters. 15-minute study: median 225
tokens, mean 233.9, max 560.

---

## 6. Verdicts, confidence, and free-text reasons

All three are present, once per game — never per turn.

| | main study | 15-minute study |
|---|---|---|
| Source table | `tt_verdict.csv` (1,141 rows) | `tt_transcripts.csv`, deduplicated to 392 games |
| Verdict values | A 589 / B 552 | A 205 / B 187 |
| `is_correct` | TRUE 605 / FALSE 536 (0.530) | TRUE 206 / FALSE 186 (0.526) |
| Confidence present | 1,141 / 1,141 | 392 / 392 |
| Confidence range | −100 … +100, median 5, sd 81.2 | −100 … +100, median 16, sd 77.9 |
| Free-text reason present | 1,141 / 1,141 | 392 / 392 |
| Reason length | median 51 chars / 9 words, max 626 chars | median 73 chars / 14 words, max 453 chars |

**Confidence is signed toward a label, not toward the chosen answer**: negative
values mean confidence in an *A* verdict and positive in a *B* verdict
(codebook). Converting to a probability that the *chosen* witness is human
requires `|confidence|`, and any calibration analysis must do that conversion
explicitly — the raw column has mean ≈ −1, which is a mid-scale artefact of the
sign convention, not indecision.

The main study additionally ships GPT-4o classifications of the free text
(`reason_classifications_4o.csv`, 1,500 rows over 1,141 verdicts;
`strategy_classifications_4o.csv`, 2,090 rows) against the 23-class
`3p_reason_scheme.csv` and the 26-row `tt_strategyclass.csv`. These are
model-generated labels by the original authors, not human annotation — useful as
a strategy taxonomy to seed the D0 question bank (§13.2 of the design), not as
ground truth.

---

## 7. Data-integrity findings

| Check | main study | 15-minute study |
|---|---|---|
| Duplicate `(game_id, conversation_label)` transcript rows | **2** (game 2197, both labels) | 0 |
| Games with more than one verdict | **1** (game 2197: same verdict/reason, confidence **100** and **45**) | 0 |
| Conversations with zero messages | **48** | 0 |
| Games with zero messages | **8** | 0 |
| Verdict rows whose game is missing from `tt_game` | 0 | 0 |
| Message rows whose conversation is missing | 0 | 0 |
| Games without exactly two conversations | 0 | 0 |

Game 2197 is the reason `tt_transcripts.csv` has 2,282 rows rather than 2,280
and `tt_verdict.csv` has 1,141 rather than 1,140. The two verdict rows disagree
on confidence only. **A canonical loader must deduplicate deterministically and
record which row it kept**; silently taking the first would put an arbitrary
confidence into a calibration analysis.

The 48 empty conversations (8 games with no messages at all) also confirm that
the main-study export did *not* apply the ≥2-message exclusion, unlike the
15-minute export.

---

## 8. PII and free text needing review before use

Screens are regex-based flags for human review, not a claim of cleanliness.

| Field | Non-empty values | Screen hits |
|---|---:|---|
| `tt_message_anonymized.content` (main) | 18,051 | anonymization placeholder 404; phone-like 2 |
| `tt_message_anonymized.content` (15-min) | 18,932 | anonymization placeholder 141; phone-like 1 |
| `tt_verdict.reason` | 1,141 | none |
| `tt_profile.strategy` | 307 | none |
| `tt_profile.strategy_change` | 305 | none |
| `tt_profile.other` (main) | 234 | **phone-like 1 — a real finding, see below** |
| `tt_profile.other` (15-min) | 149 | **e-mail-shaped 1** |
| `tt_profile.expt_aware_details` | 29 | none |
| `tt_witness.prompt` | 1,140 (AI witnesses only) | none |

Anonymization placeholders in the main study: `<NAME>` ×362, `<LOCATION>` ×55,
`<USERNAME>` ×7, `<OTHER_PII>` ×6, `<DOB>` ×2. 431 of 18,058 messages are marked
`is_changed = TRUE`, consistent with an LLM-performed redaction pass.

**Needs review before use:**

1. **A residual re-identifier.** One `tt_profile.other` free-text response
   contains what appears to be the respondent's own Prolific participant ID (a
   24-character hex worker id). Prolific ids link across studies. The value is
   not reproduced here. **Exclude `tt_profile.other` from derived artefacts
   until reviewed; consider notifying the authors** (registry §14, item 5).
2. One 15-minute-study `tt_profile.other` response trips the e-mail screen.
3. The two phone-like message hits are false positives (a date range and a
   counting sequence) — verified by reading them.
4. Human-witness identity columns (`name`, `city`, `country`, `languages`,
   `timezone`) are `NA` throughout; those fields carry prompt context for AI
   witnesses only.
5. Message content is unscreened for sensitive topics; participants discuss
   personal details as a detection strategy. No systematic pass has been run.

---

## 9. What Inverse Turing Bench's "length ≥ 50" measures

Determined by inspecting the released artifact, per design §18 Stage B step 2 —
pulled forward because it decides how much of the corpus a reproduction covers.
Script: `v2/scripts/itb_length_unit.py`; output: `itb_length_unit.json`.

The benchmark's released file
(`InverseTuringBench_o50_conversations_shuffled.csv`, from the authors' HF Space
`roc-hci/Inverse-Turing-Bench-Leaderboard`; **not committed here** — it is a
separate source and needs its own registry record) has **557 rows** and three
columns: `id`, `dialogueA`, `dialogueB`. The dialogue strings are verbatim
copies of the `tt_transcripts.transcript` column, including the `I: ` / `W: `
line prefixes.

Matching every released dialogue back to its source transcript by
whitespace-normalised string equality recovers **557 games**, with 0 unmatched
sides (one side matched only after case-folding: a pandas `nan` vs `NaN` in an
empty witness message, game 3009) and 0 pairs spanning two games — so a released
"pair" is exactly one game's two conversations.

Applying "≥ 50, both sides of the pair" under each candidate unit:

| Candidate unit | Games selected | = 557? | Set identical to the release? | Released dialogues below 50 |
|---|---:|---|---|---:|
| **whitespace tokens, `I:`/`W:` prefixes included** | **557** | **yes** | **yes** | **0** |
| whitespace tokens, prefixes stripped | 464 | no | no (93 released games excluded) | 102 |
| turns (messages) | 0 | no | no | 1,114 (all) |
| characters | 1,037 | no | no (480 extra) | 0 |

**Answer: the unit is whitespace-delimited tokens of the released transcript
string with the `I: ` / `W: ` role prefixes counted, applied to both dialogues
of a pair** — i.e. `len(transcript.split()) >= 50` on the OSF
`tt_transcripts.transcript` column. The identification is exact on both tests:
the count matches (557) and the selected game set is identical to the set
present in the release, and the minimum token count among released dialogues is
exactly 50.

It is **not** turns: the longest conversation in the corpus is 39 messages, so a
turn-based filter at 50 would select nothing. It is not characters (that would
keep 1,037 games), and it is not word count with prefixes removed (102 released
dialogues fall below 50 on that measure).

**Consequence for Track A:** the ITB reproduction (arm A1) covers 557 of 1,140
games — **48.9%** of the main study, biased toward longer conversations (mean
90.5 tokens per released dialogue against 65.0 across the corpus). Any
reproduction must report that coverage, and any comparison between our
full-corpus numbers and ITB's numbers must account for it. The same filter on
the 15-minute study would keep 373 of 392 games (95.2%), because those
conversations are three times longer.

---

## 10. What is MISSING relative to v2.1's Track A needs

Present and sufficient:

- Real human-vs-AI paired dialogue with ground-truth labels (`is_human`).
- Distinguishable interrogator/witness turns, with timestamps and order.
- A final verdict, a signed confidence, and a free-text reason per game.
- Stable person-level identifiers for split-by-participant (`user_id`).
- Full AI system prompts, so persona/no-persona is a recoverable factor.
- A published reproduction target (ITB) with a now-known selection rule.

Missing, and the consequence for each:

1. **Any turn-level belief.** Confidence exists once, at the end. The message
   table has no per-turn probability, and nothing corresponds to the ledger of
   design §7.1. Turn-wise evaluation (arm A3) therefore scores *our* estimator's
   trajectory against a fixed final label — there is no human turn-wise
   trajectory to compare against or to imitate. D1 imitation of human belief
   updating is not available from this corpus at all.
2. **Counterfactual questions.** The corpus is fixed. It cannot say what would
   have happened had a different question been asked, which is precisely why
   v2.1 keeps the active track inside D0 and why replay (Stage D) can only test
   belief updating.
3. **Model-family breadth.** Two substantive families plus ELIZA in the main
   study. Leave-one-family-out is a two-point exploratory contrast, not evidence
   of cross-family generalisation. The 15-minute study adds GPT-5 only with a
   study-level confound.
4. **Ordinary (non-adversarial) human conversation.** The human class is people
   *trying to prove they are human under interrogation*. Nothing in this corpus
   supports claims about unsuspecting conversation — the design already forbids
   that claim (§4) and the data confirm there is no alternative sample here.
5. **A temporally clean representation model is not supplied.** Witness
   conversations were generated in March 2025 (and January 2026); the frozen
   representation checkpoint for arms A2/A3 must predate that, and choosing one
   is still an open decision (design §19). The corpus itself gives no help here.
6. **Per-turn latency semantics are contaminated by design.** AI messages were
   emitted with an artificial per-character delay (`secs_per_char` in
   `tt_aimodel.csv`), so timestamp-derived features are partly a property of the
   experiment harness, not of the writer. Any timing feature is a
   collection-artifact candidate for the template-normalisation analysis.
7. **No held-out final test partition is defined by the release.** Splitting by
   `user_id`, with same-participant games kept together and the ITB 557 kept
   identifiable, is ours to define and freeze before any adapter comparison.
8. **The published analysed subset (1,023 games) is not reproducible from the
   stated criteria** (§3). Until the `.Rmd` scripts are run, our numbers and the
   paper's numbers are not directly comparable.
9. **Human witness text is scarce per game in the main study** — median 6
   witness messages, 7.2 messages per human conversation. Turn-wise calibration
   curves in the main study will be short; the 15-minute study (median 18) is
   the better substrate if its consent status can be confirmed (registry §14,
   item 3).

---

## 11. Reproducing this report

```bash
make v2-fetch-3p     # download OSF jk7bw into v2/data/sources/ + MANIFEST.json
make v2-inspect-3p   # measure both studies -> v2/results/stage_a/*.json
```

The ITB determination needs the benchmark file, which is not committed:

```bash
venv/bin/python v2/scripts/itb_length_unit.py --itb-csv /path/to/InverseTuringBench_o50_conversations_shuffled.csv
```

`make v2-fetch-3p` is idempotent: it never overwrites an existing file, it
re-hashes what is on disk, and it fails loudly if a hash stops matching OSF.
`tests/test_v2_sources.py` re-verifies every recorded hash as part of
`make test`, so drift in the immutable layer surfaces as a test failure.

**This repo is public, so the corpus is not committed.** `.gitignore` excludes
everything under `v2/data/sources/` except `MANIFEST.json` and the registry
records — the hashes and provenance are published, the text is not (registry §5;
design §8.2's rule for the unclear-redistribution case). `v2/data/canonical/`
and `v2/data/sft/` are gitignored for the same reason.

Constraints honoured this session: the downloaded files were opened read-only;
no `v2/data/canonical/` or `v2/data/sft/` data was written; no training or model
inference was run.
