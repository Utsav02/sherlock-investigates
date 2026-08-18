# Author contact — drafts, NOT SENT

**Date drafted:** 2026-08-17 · **Status: unsent.** Nothing in this file has been
sent to anyone. Record the send date and any reply in
`v2/data/sources/registry/jones_bergen_2025.md` §14 (items 1, 3, 5) when that changes.

**Recipient:** `cameron@ucsd.edu` — the contact given in both `data/codebook.txt`
and `15_mins/codebook.txt`, and the sole bibliographic contributor on OSF node
`jk7bw`.

Two separate emails rather than one. Email B reports residual participant PII;
folding that into a licensing question would bury it, and it deserves its own
subject line and its own reply. Send B first if only one goes.

---

## Email A — 15-minute study: consent scope and licence

**Subject:** OSF jk7bw — licence, and consent scope for the `15_mins` release

> Dear Dr Jones,
>
> I'm a researcher working on AI-versus-human dialogue detection, using the
> three-party Turing test transcripts you released on OSF (`jk7bw`) as an
> evaluation corpus. Thank you for publishing the full data and analysis code —
> being able to run your `.Rmd` scripts directly meant I could reproduce the
> analysed subset of 1,023 games exactly, rather than guessing at the exclusion
> criteria.
>
> I have two questions I could not resolve from the release itself.
>
> **1. The `15_mins/` directory.** It appears to be a separate 15-minute study
> run in January 2026, with a GPT-5 witness, and I could not find an accompanying
> paper or preregistration. Its codebook documents the same anonymisation and
> exclusion pipeline as the 2025 study, but I don't want to assume that the
> consent language in arXiv:2503.23674 — participants consenting for anonymised
> data including transcripts to be used for analysis and shared publicly — also
> covers that collection. Could you confirm whether the 15-minute study carries
> the same participant consent and IRB approval, and whether there is a paper or
> preregistration I should be citing?
>
> **2. Licence.** The OSF node does not declare a licence, and I would rather ask
> than assume. Would you be willing to state one for the data?
>
> If a general licence is more than you want to commit to, the specific question
> I need answered is narrower: I plan to use the transcripts **only for
> evaluation** — reproducing passive-detection baselines and measuring
> calibration — and I do not intend to fine-tune any generative model on them. I
> ask because the Inverse Turing Bench paper, which you co-authored, names
> training models to be less detectable as a risk of that data, and I've treated
> that as a constraint on my own use. If you'd prefer that this corpus not be
> used for model training at all, I'll record that and abide by it; if
> evaluation-only use is fine with you, confirming that would let me proceed
> without over-restricting myself.
>
> I'm not redistributing any transcript text. My repository publishes only
> scripts, file hashes, and aggregate statistics.
>
> Many thanks for your time,
>
> Utsav Singh
> singh.utsav02@gmail.com

---

## Email B — residual participant identifiers in `tt_profile.other`

**Subject:** OSF jk7bw — two participant IDs remaining in `tt_profile.other`

> Dear Dr Jones,
>
> While screening the OSF `jk7bw` release before use, I found what look like two
> residual participant identifiers that the anonymisation pass may have missed.
> I'm reporting them rather than acting on them, and I have not reproduced the
> values anywhere.
>
> Both are in the free-text `other` column of `tt_profile.csv` — one row in
> `data/`, one row in `15_mins/`. Each contains a 24-character hexadecimal string
> of the shape of a Prolific worker ID. In the `15_mins/` row the same string also
> appears as the local part of an address at `email.prolific.com`, i.e. the
> participant's Prolific relay address.
>
> I mention it because the codebook describes stripping PII *columns* from the
> profile table, which wouldn't catch something a participant typed into a
> free-text box; and because Prolific IDs are linkable across studies, so these
> are re-identifiers rather than just stray text. Each affected row is a single
> value out of 234 and 149 non-empty responses respectively, so this looks like an
> edge case in the pass rather than a systematic problem.
>
> On my side I've excluded the `tt_profile.other` column entirely from anything
> derived from the corpus, and I haven't recorded or transmitted the values. I'm
> happy to share the two `user_id`s privately so you can locate the rows, if that
> would help — just let me know and I'll send them directly rather than putting
> them in this thread.
>
> Best wishes,
>
> Utsav Singh
> singh.utsav02@gmail.com

---

## Notes on the drafts

- **Email B deliberately does not contain the values, the `user_id`s, or the
  regexes that found them**, and offers the identifiers separately on request.
  A vulnerability report should not itself be the disclosure.
- **Email A's second question is framed to be answerable.** Asking "please
  license your data" invites no reply; asking "is evaluation-only use acceptable,
  and would you prefer no training use at all?" is a yes/no a busy author can
  answer in one line, and either answer resolves registry §12 for us.
- **Neither email asks for anything the project needs in order to proceed.**
  Track A's required path (§4 below) does not depend on a reply.

## If there is no reply: Track A degrades to the 5-minute corpus alone

Planned so that a non-reply costs coverage, not the study. Nothing below needs
`15_mins`.

| Track A element | With `15_mins` cleared | Without it (**the default plan**) |
|---|---|---|
| Primary estimator + calibration | unchanged | unchanged — cross-fitted over the frozen `main_study_v1` split, 911 development games |
| Gate 1 rung 1 (held-out people) | unchanged | unchanged |
| Gate 1 rung 2 (held-out population) | unchanged | unchanged — SONA ↔ Prolific, 500/640 games, participant pools are disjoint by construction |
| Gate 1 rung 3 (collection-artifact removal) | unchanged | unchanged |
| Gate 1 rung 6 (held-out **dataset**) | runs, three stages | **not run**; every downstream claim must state that no dataset-level holdout was performed |
| GPT-5 / post-2025 model generation | available | **not available**; the newest witness is GPT-4.5 (Feb 2025) and generalization to later models is untested, not merely unproven |
| Turn-wise calibration substrate | 15-min study preferred (median 22 messages/conversation) | main study only (median 8) — turn-wise curves are short and must be reported as such |

The two real costs of a non-reply are the **dataset-level holdout** and the
**longer conversations**. Both are limitations to state, not blockers: Gate 1 as
proposed in `gate1_out_of_source.md` §5 is decidable on rungs 1–3, all of which
use only the cleared main-study corpus.

**Standing rule while unresolved:** `15_mins` stays at inspection-only. It is
already excluded from the frozen split (held whole, unpartitioned), so no code
path can pull it into training or evaluation by accident.
