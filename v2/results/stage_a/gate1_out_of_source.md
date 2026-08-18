# Gate 1's "out-of-source" requirement: what it can mean on this corpus

**Date:** 2026-08-17 · **Stage:** A · **Status:** proposal. `v2/experiment_design.md`
is **not** edited by this document; §5 below is the exact replacement text to
apply when the owner accepts it.

Precision numbers are from `v2/scripts/precision_track_a.py`; structural counts
are measured by `v2/scripts/build_splits.py` and the Stage A inspection.

---

## 1. The problem

Gate 1 currently reads, in full:

> **Gate 1: Track A real-passive signal**
> A passive detector must show useful out-of-source discrimination and calibration.
> Strong within-dataset performance with failed cross-source transfer is a stop or
> redesign result, not permission to fine-tune.

"Out-of-source" is never defined, and three things are wrong with leaving it that way.

**(a) It does not match the design's own ladder.** RQ1 already specifies something
more careful: *"held-out people, prompts, sources, or model families as the data
permits."* Gate 1 collapses that four-rung ladder to one rung and drops the "as the
data permits" qualifier that makes it executable.

**(b) Exactly one out-of-source option exists, and it is not cleared.** Every
source in §8.1 other than the three-party corpus is at "verify terms" status and
none is downloaded. The only other body of real transcripts in hand is the
15-minute study inside the same OSF node — whose Gate 0 is explicitly unresolved
(registry §14 item 3: no paper, no preregistration, consent unconfirmed, no
declared licence). **A gate that can only be evaluated by using data we have not
been permitted to use is not a gate.**

**(c) A gate must be able to fail informatively.** Gate 1's stated consequence is
"stop or redesign". A stop decision has to be attributable, and as §3 shows, the
cross-study comparison confounds at least six axes at once.

---

## 2. Is 5-minute-develop → 15-minute-evaluate the right instantiation?

**It is a legitimate test and the wrong primary one.** Three reasons, in order of
weight:

1. **It is permission-blocked.** Gate 0 for `15_mins` is unresolved. Making the
   project's second gate depend on an unanswered email means Track A can be
   stalled by a non-reply. §4 of this document is the graceful-degradation plan;
   §5's wording is written so the gate never depends on it.

2. **A failure would be unattributable.** See §3. Study, date, model generation,
   conversation length, recruitment population, and export pipeline all move
   together. If transfer fails, the honest report is "it failed for one of six
   reasons" — which cannot justify "stop or redesign".

3. **It is the least precise option available.** Restricted to the witness systems
   the two studies share — the only version that controls model generation — it
   has 244 games and MDD 9.4pp, against 6.3pp for the within-corpus population
   holdout. The most confounded test is also the weakest.

Against that, it has one real virtue that nothing else has: it is the only
available test that moves the **dataset**, and it contains **GPT-5**, a model
generation that postdates every witness in the development corpus. That is the
strongest generalization evidence obtainable here — which is why the proposal
below keeps it as a named, conditional, *promotable* rung rather than discarding it.

---

## 3. What 5-min → 15-min confounds

Measured, not assumed. Every row differs between the two studies.

| Axis | Main study (`data/`) | 15-minute study (`15_mins/`) |
|---|---|---|
| **Study / export** | pre-exclusion export, 1,140 games | export with analysis exclusions already applied, 392 games |
| **Date** | 2025-03-06 → 03-14 | 2026-01-13 → 01-14 |
| **Model generation** | GPT-4.5, GPT-4o, Llama-405B, ELIZA | **GPT-5**, Llama-405B, ELIZA |
| **Conversation length** | median 8 messages/conversation | median 22 (AI 28, human 18) |
| **Recruitment** | Prolific (183) + SONA/UCSD (140) | Prolific only (227) |
| **Message cap** | 300-char cap holds (2 rows at 303) | cap violated 21× (max 676 chars) |
| **Verdict table** | `tt_verdict.csv` present | **absent**; verdict survives only on transcript rows |
| **Participants** | 323 users | 227 users, **zero overlap** with the main study |

The user-disjointness is the one axis that is a *feature*: it makes any cross-study
result a genuine held-out-people test. (Caveat in §6.)

### What can be deconfounded, and what cannot

Two structural facts, both measured this session, make partial deconfounding possible:

**Shared witness systems.** `eliza` and `llama-405b_quinn` appear in *both*
studies — 422 games in the main study, 244 in the 15-minute study. Restricting the
cross-study test to those two systems **removes the model-generation axis**,
leaving study/date/length/recruitment.

**Length is controllable.** 15-minute transcripts can be truncated to the main
study's message-count distribution, **removing the length axis** as well. What
remains after both controls is study ∧ date ∧ recruitment — three axes that
cannot be separated by any analysis, because no data varies them independently.

**Conversely**, within the 15-minute study alone, GPT-5 (148 games) versus
Llama-405B (149 games) isolates model generation at fixed study, date, and length.

So the cross-study comparison decomposes into interpretable pieces. It just cannot
be run as one number and read as one conclusion.

---

## 4. What is available unconditionally

The decisive structural finding: **recruitment population is perfectly nested
inside the participant co-occurrence components — 0 of 15 components mix SONA and
Prolific participants.** SONA (500 games) and Prolific (640 games) are entirely
separate participant pools collected in the same window with the same apparatus,
the same witness systems, and the same length distribution.

That yields a clean population-transfer test needing no new permission:

| Rung | Test | n (eval) | MDD @ ICC 0.10 | Needs Gate 0 on `15_mins`? |
|---|---|---:|---:|---|
| **People + population** | train SONA → eval Prolific | 640 | **6.3pp** | no |
| | train Prolific → eval SONA | 500 | **7.1pp** | no |
| **Prompts** | train minimal-prompt → eval persona | 561 | 6.7pp | no |
| | train persona → eval minimal-prompt | 432 | 7.6pp | no |
| **Model family** | train OpenAI → eval Llama | 412 | 7.8pp | no (but 2 families only) |
| | train Llama → eval OpenAI | 581 | 6.6pp | no (but 2 families only) |
| **Dataset (model-controlled)** | main → 15-min, shared systems, length-matched | 244 | 9.4pp | **yes** |
| **Dataset (full)** | main → 15-min, all systems incl. GPT-5 | 392 | 7.4pp | **yes** |

**Naming these honestly matters.** The SONA↔Prolific test is *not* out-of-source:
it is out-of-population within one collection apparatus, and it cannot detect a
collection artifact shared by both halves. Calling it a source holdout would be
exactly the kind of overclaim the design's §4 exists to prevent. The proposed
wording therefore calls each rung what it is.

---

## 5. Proposed replacement wording for §16, Gate 1

> ### Gate 1: Track A real-passive signal
>
> A passive detector must show useful discrimination and calibration that
> survives held-out entities, and the strongest holdout the cleared data
> permits. Strong within-dataset performance that fails to transfer is a stop or
> redesign result, not permission to fine-tune.
>
> **Gate 1 is evaluated on a ladder, and each rung is named for what it actually
> holds out.** The rungs below are ordered by strength of evidence, not by
> convenience. Every number is reported with clustered 95% intervals on the
> participant, and with the pre-registered contrast set frozen in
> `v2/results/stage_a/precision_and_contrasts.md`.
>
> **Required to pass Gate 1** (all three; none needs data beyond the cleared
> main-study corpus):
>
> 1. **Held-out people.** Leave-one-component-out cross-fitted discrimination and
>    out-of-fold calibration, using the frozen split `main_study_v1`. No game is
>    scored by a model that saw either of its participants.
> 2. **Held-out population.** Train on one recruitment population and evaluate on
>    the other (SONA/UCSD ↔ Prolific), in both directions. These pools are
>    disjoint in participants by construction — recruitment source is perfectly
>    nested inside the co-occurrence components — but they share a collection
>    apparatus, so this is a population holdout and must not be described as a
>    source or dataset holdout.
> 3. **Collection-artifact removal.** Report discrimination before and after
>    template normalization. A signal that does not survive normalization is a
>    collection artifact and fails Gate 1 regardless of rungs 1 and 2.
>
> **Reported but not gating** (insufficient support to carry a pass/fail):
>
> 4. **Held-out prompts.** Persona ↔ no-persona transfer, both directions.
> 5. **Held-out model family.** OpenAI ↔ Meta-Llama transfer. With exactly two
>    substantive families this is a two-point comparison and is exploratory by
>    §14; it may not be cited as cross-family generalization.
>
> **Conditional strongest rung — held-out dataset.** If and only if Gate 0 for
> the 15-minute study is resolved affirmatively, run the cross-study transfer in
> three stages and report all three:
>
> 6a. restricted to the witness systems both studies share (ELIZA, Llama-405B)
>     **and** length-matched to the main study's message-count distribution —
>     this controls model generation and conversation length;
> 6b. restricted to the shared systems, without length matching — the difference
>     from 6a isolates the length axis;
> 6c. unrestricted, including the GPT-5 witness — the strongest generalization
>     evidence available, and the least attributable.
>
> A failure at 6c alone, with 6a passing, is a **length or model-generation**
> finding and is not grounds to stop. A failure at 6a is a genuine dataset-transfer
> failure and is grounds to stop or redesign. If Gate 0 is unresolved or negative,
> Gate 1 is decided on rungs 1–3 alone, and **every downstream claim must state
> that no dataset-level holdout was performed.**
>
> Gate 1 is not passed by a numerically significant result on any single rung. It
> requires rungs 1–3 together, and the write-up must state which rungs were run,
> which were unavailable, and what each one confounds.

---

## 6. Caveats recorded with the proposal

1. **Zero cross-study user overlap is a property of the identifiers, not of the
   humans.** The two studies use entirely disjoint `user_id` ranges (main
   2448–3866; 15-minute 109818–110649), which shows the releases were numbered
   separately. Both drew on Prolific ten months apart, so the same worker could
   have taken part in both under two different ids and nothing in the release
   would reveal it. Treat "held-out people" across studies as *very likely* but
   not verified; the within-study holdouts (rungs 1 and 2) are verified.
2. **Rung 2 cannot detect a collection artifact shared by both populations.**
   That is what rung 3 is for, and why rung 3 is required rather than optional.
3. **The 15-minute study's export already applied analysis exclusions** while the
   main study's did not. Cross-study evaluation must therefore either apply the
   main study's exclusions first, or report that the two sides were filtered
   differently.
4. **These MDDs assume paired discordance 0.25.** Transfer comparisons are between
   a model and a baseline on the same evaluation games, so the pairing holds, but
   the discordance figure is a guess until estimators exist.
