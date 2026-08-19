# Stage A step 1 — literature matrix

**Date:** 2026-08-18 · Discharges (partially — see the honesty note) §18 Stage A
step 1: *"Build the literature matrix: question, population, data, intervention,
baselines, metrics, limitations, licences, and stated gaps."*

---

## ⚠️ Honesty note — read before using any cell

**This matrix is complete for what the repository can support without network
access, and no further.** No paper was downloaded or read while writing it; that
was a session constraint.

Every cell is therefore marked:

| mark | meaning |
|---|---|
| **[R]** | **from a repository artifact** — the registry, the data inspection, or a measured number. Trustworthy. |
| **[P]** | from the paper as **quoted in our own registry** (JB 2025 only). Trustworthy as a quotation. |
| **[U]** | **UNVERIFIED.** Background knowledge or the design doc's own one-line description. **Must be confirmed against the paper before being cited.** |

**Do not quote a [U] cell in any writeup.** Several of these papers postdate the
assistant's training data (e.g. Inverse Turing Bench, arXiv 2606.21844, June 2026;
ConfTuner 2508.18847), so for those even the topic line is inference from the
design doc's own description, not knowledge of the work.

**To finish this properly** requires fetching ~15 papers and filling the [U]
cells. That is a bounded task (one session, network access, no compute) and it is
the remaining half of Stage A step 1.

---

## Group 1 — Turing-test / human-vs-AI detection (directly load-bearing for Track A)

| paper | question | population | data | intervention | baselines | metrics | limitations | licence | stated gaps |
|---|---|---|---|---|---|---|---|---|---|
| **Jones & Bergen 2025**, *LLMs Pass the Turing Test* (2503.23674) | can people distinguish an LLM from a human in a 3-party test? **[P]** | 1,023 analysed games / 284 participants **[P]**; release holds **1,140 / 323** **[R]** | our primary corpus; 5-min games, median 8 msgs/conv **[R]** | witness system + persona prompt **[R]** | ELIZA as manipulation check **[R]** | interrogator accuracy; measured 0.530 overall **[R]** | adversarial humans, not ordinary conversation; 2 substantive families **[R]** | **none declared**; consent covers analysis + public sharing **[P]** | no turn-level belief; fixed transcripts **[R]** |
| **Jones & Bergen 2025**, 15-minute study (same OSF node) | as above at 15 min **[R]** | 392 games / 227 participants **[R]** | median 22 msgs/conv; **GPT-5** witness **[R]** | longer time limit **[R]** | ELIZA **[R]** | interrogator accuracy 0.526 **[R]** | no located paper or preregistration **[R]** | **Gate 0 UNRESOLVED — unread** **[R]** | consent status unconfirmed **[R]** |
| **Jones & Bergen 2024** (2405.08007) | can people distinguish GPT-4 from a human? **[U]** | ~402 retained games **[U]** | two-party design **[U]** | GPT-3.5 / GPT-4 / ELIZA **[U]** | ELIZA **[U]** | pass rate **[U]** | two-party ≠ three-party **[U]** | unverified **[U]** | no turn-level ledger **[U]** |
| **Inverse Turing Bench** (2606.21844) | can *models* identify the human in a dialogue pair? **[U]** | 557 pairs derived from JB 2025 **[R]** | our reconstruction of its filter is exact: whitespace tokens incl. `I:`/`W:` prefixes, ≥50, both sides **[R]** | zero-shot judging **[U]** | GPTZero, frontier LLMs **[U]** | reported GPTZero 89.4%, Claude Opus 4.6 77.9% **[U]** | 48.9% corpus coverage, biased long (90.5 vs 65.0 tokens) **[R]** | benchmark terms unverified; **no registry record** **[R]** | authors caution against training for undetectability **[R]** |
| **Human or Not?** (2305.20010) | large-scale gamified Turing test **[U]** | — **[U]** | — **[U]** | — **[U]** | — **[U]** | — **[U]** | older models; raw availability unclear **[U]** | unverified **[U]** | — **[U]** |
| **SPADE** (ACL 2025 LLMSEC) | detecting LLM-generated dialogue **[U]** | — **[U]** | mainly synthetic customer-service (MultiWOZ-derived) **[U]** | — **[U]** | — **[U]** | — **[U]** | not adaptive interrogation **[U]** | Apache label unverified; upstream MultiWOZ terms unverified **[U]** | — **[U]** |
| **HC3** | human vs ChatGPT answer detection **[U]** | — **[U]** | static QA pairs **[U]** | — **[U]** | — **[U]** | — **[U]** | not dialogue; single-model signal **[U]** | unverified **[U]** | — **[U]** |
| **HANSEN** | human vs LLM spoken-text detection **[U]** | — **[U]** | spoken-text corpora **[U]** | — **[U]** | — **[U]** | — **[U]** | spoken domains, not interrogation **[U]** | per-source terms unverified **[U]** | — **[U]** |

**Track A's own result belongs in this group and is [R] throughout:** two methods
(TF-IDF; frozen Qwen2.5-7B + logistic head) reach 0.960 / 0.866 paired accuracy
against seen respondent configurations and both fail to transfer across persona
prompt families (0.604/0.493 and 0.495/0.448). See
`v2/results/track_a/corrections_20260818_213000.md` for the claim-level wording.

---

## Group 2 — active information seeking (load-bearing for Track B / Stage C)

| paper | question | intervention | baselines | metrics | limitations | stated gaps |
|---|---|---|---|---|---|---|
| **Uncertainty of Thoughts** (2402.03271) | can an LLM choose questions that maximise expected information gain? **[U]** | UoT planning over a question bank **[U]** | random / fixed order **[U]** | success rate, questions-to-solve **[U]** | simulator-bound **[U]** | — **[U]** |

**This group has one entry, and that is a gap, not a summary.** Stage C's design
names UoT as arm B2 — the *only* published active baseline the plan commits to.
Before building D0, this group needs the surrounding literature (20-questions /
active-inference / information-gain dialogue agents) so that B2 is a considered
choice rather than the single method that happened to be cited. **This is the
highest-value remaining piece of the matrix**, because it is the one that
constrains work not yet done.

---

## Group 3 — calibration and confidence training (relevant to the re-anchored P1)

| paper | question | intervention | limitations | why it matters here |
|---|---|---|---|---|
| **ConfTuner** (2508.18847) | train calibrated verbal confidence **[U]** | proper-scoring / tokenized-Brier style training **[U]** | **[U]** | the named method for the deferred proper-scoring arm (§13.3) |
| **LACIE** (2405.21028) | listener-aware calibration of expressed confidence **[U]** | preference training on confidence **[U]** | **[U]** | alternative to ConfTuner for the same arm |

Directly relevant now: P1 was re-anchored from discrimination to calibration
(2026-08-18), and A2 was found to be a probabilistic head with **no** calibrator.
If a nested calibrator is ever added, this group is the prior art to consult
first.

---

## Group 4 — chain-of-thought faithfulness and monitoring (frames §7.2)

| paper | claim as used in our design | mark |
|---|---|---|
| Measuring Faithfulness in CoT (2307.13702) | printed CoT is not necessarily the computation | **[U]** |
| LMs Don't Always Say What They Think (2305.04388) | CoT can rationalise post hoc | **[U]** |
| Monitoring Reasoning Models / Obfuscation (2503.11926) | optimising CoT to look acceptable degrades it as a monitor | **[U]** |
| CoT Monitorability (2507.11473) | monitorability is real but fragile | **[U]** |
| Emergent Misalignment (2502.17424) | narrow finetuning can produce broad behaviour change | **[U]** |
| Sleeper Agents (2401.05566) | trained behaviours can persist through safety training | **[U]** |
| LIMA (2305.11206) | small high-quality sets can suffice for alignment-style SFT | **[U]** |

These motivate §4's list of claims the project will not make and §7.2's decision
to keep CoT as a monitoring channel rather than a training target. They are
**not** load-bearing for any current measurement, which is why they are lowest
priority to verify.

---

## Gap list — what finishing Stage A step 1 requires

1. **Fetch and verify Group 2 first.** One entry is not a literature review, and
   Stage C depends on it. Confirm UoT's actual baselines and metrics, and find the
   adjacent active-questioning work the design does not cite.
2. **Fetch Inverse Turing Bench** (2606.21844) and give it a §8.2 registry record.
   It is currently cited for two numbers used in comparison (GPTZero 89.4%, Claude
   Opus 4.6 77.9%) that are **[U]** — those comparisons should not appear in any
   external writeup until verified. The corrections sheet already retires the
   comparison; this would let it be restated correctly or dropped for good.
3. **Fetch JB 2024** (2405.08007) — cheapest replication candidate, same authors.
4. **Fill Group 3** before any calibration work.
5. **Group 4 last** — framing only, nothing measured depends on it.

Items 1–3 are the ones that would change a decision. Items 4–5 would not.
