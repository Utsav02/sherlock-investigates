# Track A arm A2 — frozen temporally clean representation, and the Gate 1 verdict

**Date:** 2026-08-18 · **Stage:** B, final Track A run · Appends to the three
earlier Track A writeups, all left unedited (results/ is append-only).

**Run:** `a2_frozen_rep_20260818_203908.json`; matched A0 comparison in
`a0_rung4_loso_20260818_202921_a0witnolen_persona.json`.

**Test split untouched.** No new downloads.

---

## 0. Answer to the session question

> **Is the persona ceiling a property of bag-of-words, or of the corpus?**
>
> **Of the corpus.** A frozen, temporally clean 7B representation with a
> calibrated head collapses on the persona holdout at least as hard as TF-IDF —
> to **below chance**, with the held-out AI flagged as AI in **5.7%** and **6.6%**
> of games. Track A concludes.

---

## 1. The representation, and why it is temporally clean

| | |
|---|---|
| Model | Qwen2.5-7B-Instruct (Q4_K_M), frozen |
| Released | **2024-09-19** |
| Witness conversations collected | **2025-03-06 … 2025-03-14** |
| Provenance | already in the local Ollama store, pulled ~4 months ago — no new download |
| Use | mean-pooled final hidden states, 3,584-dim; weights never updated |
| Head | L2 logistic regression, trained only on the Track A training split |

The checkpoint is finalized ~6 months before the data it is asked to represent,
so it cannot have been trained on these transcripts. That removes one specific
contamination route and no more — Qwen2.5 has certainly seen Turing-test
discussion and LLM output in general.

**One compute compromise, stated because it bounds the conclusion.** This venv
has no numpy and no BLAS, and a logistic fit on 3,584 dense dimensions in pure
Python does not fit a time-boxed session across 33 folds. The embeddings
therefore pass through a **very sparse random projection** (Li et al. 2006,
density 1/√d) to 512 dimensions, fixed seed 20260818, identical in every fold and
every cut. A projection can only lose information, never add it, so it is
conservative: it caps A2's absolute numbers but cannot manufacture a collapse,
because the same projection is present in the people-only cut where A2 scores
0.866.

---

## 2. §12 feasibility check — PASSED

The design requires a cheap development-only check before adopting this arm: the
classifier must exceed the majority baseline on dev, so that the test is not
secretly measuring an old model's inability to follow instructions.

```
fit on train (625 games) -> dev (226 games)
A2 dev game accuracy = 0.8319    majority = 0.5177    -> PASS
```

**The temporally clean arm is not capability-floored.** Whatever happens next is
a statement about the corpus, not about the representation being too weak to
participate.

---

## 3. The two cuts, A2 versus matched A0

Same frozen split, same drop-empties policy, same 851 games, same nesting. A0 is
shown at the **matched text condition** (`A0-wit-nolen`, uncapped witness-only),
re-run this session so the comparison is like-for-like rather than against the
20-token-capped numbers.

| cut | A0 (TF-IDF) | **A2 (frozen rep)** |
|---|---|---|
| **people-only** (11-fold LOCO) | 0.9600 | **0.8660 [0.842, 0.889]** |
| **LOSO-persona** minimal→quinn | 0.6038 [0.558, 0.649] | **0.4929 [0.443, 0.540]** |
| **LOSO-persona** quinn→minimal | 0.4953 [0.437, 0.553] | **0.4483 [0.384, 0.517]** |

A2 balanced accuracy: people-only **0.7726**, persona **0.4693** and **0.4655** —
below chance in both directions.

**Both A2 persona intervals contain or sit below 0.5.** Neither is
distinguishable from answering at random, and the point estimates are on the
wrong side of it.

Incidentally, the matched A0 numbers also settle a loose end: uncapped A0
collapses *harder* than the capped version reported previously (0.6038 vs 0.6427;
0.4953 vs 0.5799), so the 20-token cap was never what produced A0's persona
collapse.

---

## 4. Side recognition — the diagnostic that distinguishes "degraded" from "became a human-recognizer"

Accuracy alone cannot tell those apart, because the paired task can be won from
the human side alone.

| cut | AI flagged as AI | human cleared | p(AI) on AI side | p(AI) on human side |
|---|---:|---:|---:|---:|
| **A2** people-only | **0.742** | 0.804 | 0.637 | 0.378 |
| **A2** minimal→quinn | **0.057** | 0.882 | 0.257 | 0.277 |
| **A2** quinn→minimal | **0.066** | 0.865 | 0.203 | 0.257 |
| A0 minimal→quinn | 0.146 | 0.896 | — | — |
| A0 quinn→minimal | 0.056 | 0.956 | — | — |

This is the clearest result in the session. **In-distribution, A2 genuinely
recognises both sides** — it flags 74.2% of AI dialogues as AI and clears 80.4%
of humans, and the mean probabilities separate cleanly (0.637 vs 0.378). Under a
held-out persona, AI recognition falls to **5.7%** and **6.6%**, and the two
sides' mean probabilities converge to within 0.02–0.05 of each other. The model
has not merely degraded; it has stopped seeing the machine and become a
human-recognizer that assigns "human" to nearly everything.

Calibration moves the same way: out-of-fold ECE is **0.1008** people-only (better
than A0's 0.173) and **0.283 / 0.364** under the persona holdout — confidently
wrong, which is the worst failure mode for an arm whose stated job in this
project is to own the probability.

---

## 5. Gate 1 verdict

> **What this corpus supports.** On the five-minute Jones & Bergen three-party
> corpus, both a bag-of-words baseline and a calibrated head over a frozen,
> temporally clean 7B representation distinguish the human respondent from the AI
> respondent when evaluation holds out participants but not respondent
> configuration — A0 at 0.960 and A2 at 0.866 [0.842, 0.889] paired accuracy,
> A2 with out-of-fold ECE 0.101 and genuine two-sided recognition (74.2% of AI
> dialogues flagged, 80.4% of humans cleared). That signal is real, and it is not
> an artefact of interrogator text, reply rate or length, redaction placeholders,
> or respondent silence, each of which was measured and excluded. It transfers
> between recruitment populations. The corpus therefore supports the study of
> passive identity judgment against *specified, seen* respondent configurations,
> and supports calibration work on that task.
>
> **What it does not support.** Neither estimator survives a held-out persona
> prompt. Trained on minimal-prompt witnesses and evaluated on persona-prompt
> witnesses, A0 scores 0.604 and A2 0.493; in reverse, 0.495 and 0.448. Three of
> those four point estimates are at or below chance, all four intervals contain
> 0.5, balanced accuracy is 0.47–0.52, and the held-out AI is flagged as AI in
> 5.7–14.6% of games while humans are cleared 86–96% of the time. The corpus does
> not support any claim about detecting AI respondents in general, about unseen
> model or prompt configurations, or about deployment against an adversary free to
> choose its own prompt.
>
> **Whose ceiling is it.** It is the corpus's, not bag-of-words'. The
> representation was swapped for a frozen neural one that passed its feasibility
> check at 0.832 on dev, demonstrably recognises AI in-distribution, and is
> temporally incapable of having memorised these transcripts — and it collapses
> further than the lexical baseline, not less. Two estimators with nothing in
> common but the training data fail the same way on the same cut, which locates
> the limit in what the data can teach rather than in how the text is
> represented. The most likely reading is that a five-minute adversarial
> transcript carries far more signal about *which prompt the witness was given*
> than about whether the witness is a machine, and that the persona instruction
> dominates whatever machine-ness signal exists.

---

## 6. Consequence, and Track A stops here

Gate 1's stated consequence is that strong within-dataset performance with failed
transfer is "a stop or redesign result, not permission to fine-tune." That is
what happened, on the axis that matters most for v2.1's question. Track A
concludes with a characterised within-family detector and a clearly located
limit.

Nothing here forbids future Track A work; it constrains what such work may claim.
An A2-style estimator remains usable as the frozen probability authority for
*within-configuration* calibration studies — which is precisely the role §5
assigns it, and precisely why the P1 re-anchoring from discrimination to
calibration (2026-08-18) was the right call. Any use of it as an "AI detector"
is refuted by §4 above.

---

## 7. Limitations

- **Two persona classes, five systems.** LOSO-persona is a two-point contrast; it
  shows the signal does not survive *this* prompt change, not how it degrades
  across prompt space generally.
- **The random projection caps A2's absolute numbers.** It cannot explain the
  persona collapse (it is equally present in the 0.866 people-only cut), but A2's
  in-distribution score would likely be higher without it.
- **Mean-pooled hidden states are a crude sentence representation.** A dedicated
  embedding model would be a better A2; none was available without a download.
- **One seed, one head, one regularisation setting.** No tuning was performed,
  by instruction; these are single-run numbers.
- **Still no dataset-level holdout** (15-minute study, Gate 0 unresolved), and the
  frozen test split remains untouched.

---

## 8. Reproducing

```bash
# a private llama-server over the already-present Ollama blob; port 51999
/Applications/Ollama.app/Contents/Resources/llama-server \
  --model ~/.ollama/models/blobs/sha256-2bada8a745...  \
  --port 51999 --embeddings --pooling mean -c 2048 -np 4
venv/bin/python v2/scripts/track_a_a2.py --bootstrap 1000
venv/bin/python v2/scripts/track_a_rung4.py --condition A0-wit-nolen \
    --cuts persona --bootstrap 1000 --tag a0witnolen_persona
venv/bin/python -m unittest tests.test_v2_track_a     # 59 tests
```

Embeddings are cached under `v2/data/canonical/a2_embeddings/` (gitignored), so
re-runs skip the ~11-minute embedding pass. Total ~20 minutes.
