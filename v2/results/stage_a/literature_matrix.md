# Stage A step 1 — literature matrix

**Date:** 2026-08-18 (built), **verified against sources 2026-08-18**
Discharges `v2/experiment_design.md` §18 Stage A step 1.

## Verification key

| mark | meaning |
|---|---|
| **[F]** | verified against the paper's **full text** this session |
| **[A]** | verified against the paper's **abstract / landing page** this session |
| **[R]** | from a repository artifact — registry, inspection, or a measured number |
| **[U]** | **still unverified** — not fetched. Do not quote. |

15 sources fetched. Abstract-level verification is not full-text verification:
an **[A]** cell is trustworthy as a summary, not as a basis for a precise claim.

> ⚠️ **Two findings below change decisions and are flagged in place: §1.1 (ITB
> independently replicates our persona result, and bounds our conclusion further
> than we did) and §2.1 (the active-questioning field has moved past UoT).**

---

## 1. Turing-test / human-vs-AI detection

| source | question | data | intervention | baselines | metrics / headline | limitations | licence |
|---|---|---|---|---|---|---|---|
| **Jones & Bergen 2025** (2503.23674) | can people distinguish an LLM from a human, 3-party? **[R]** | 1,023 analysed / release holds **1,140 games, 323 participants** **[R]** | witness system × persona prompt **[R]** | ELIZA **[R]** | interrogator accuracy; **measured 0.530** **[R]** | adversarial humans; 2 substantive families **[R]** | **none declared**; consent covers analysis + public sharing **[R]** |
| **Jones & Bergen 2025**, 15-min | as above at 15 min **[R]** | 392 games / 227 participants; **GPT-5** witness **[R]** | longer limit **[R]** | ELIZA **[R]** | accuracy 0.526 **[R]** | no paper/preregistration located **[R]** | **Gate 0 UNRESOLVED — unread** **[R]** |
| **Jones & Bergen 2024** (2405.08007) | can people distinguish GPT-4? **[A]** | **two-party**, 5-min conversations **[A]** | ELIZA, GPT-3.5, GPT-4 **[A]** | ELIZA **[A]** | **GPT-4 judged human 54%; ELIZA 22%; real humans 67%** **[A]** | n not in abstract **[A]** | CC BY 4.0 **[A]** |
| **Inverse Turing Bench** (2606.21844) | can *models* pick the human-only dialogue from a pair? **[F]** | **557 pairs, drawn from Jones & Bergen, filtered length ≥ 50** **[F]** — matches our reconstruction exactly **[R]** | zero-shot paired judging **[F]** | GPTZero, Claude Opus-4.6, GPT-5.5 **[F]** | **GPTZero 89.41%, Opus-4.6 77.92%, GPT-5.5 75.94%** **[F]** | see §1.1 **[F]** | CC BY 4.0 (paper) **[A]**; **dataset terms still unregistered** **[R]** |
| **Human or Not?** (2305.20010) | gamified Turing test at scale **[A]** | **>1.5M users**, 2-minute sessions, one month **[A]** | prompted-human AI **[A]** | — | **overall 68% correct; 60% when facing a bot** **[A]** | model unnamed; **raw-data release not stated in abstract** **[A]** | CC BY 4.0 **[A]** |
| SPADE · HC3 · HANSEN | — | — | — | — | — | — | **[U] not fetched — deprioritised, Track A concluded** |

### 1.1 ⚠️ ITB independently replicates our persona finding — and bounds our conclusion further than we did

Verified in ITB's full text (§ Limitations and results):

> "Opus-4.6, GPT-5.5, and the human judge's performance dropped when the AI
> witness was created with a persona prompt, as opposed to no persona (Opus:
> 95.80% vs. 59.04%; GPT-5.5: 93.36% vs. 57.56%; Human: 68.53% vs 39.85%),
> **while GPTZero's performance remains consistent.**"

> "both LLM and human judges are susceptible to carefully crafted prompt
> personas, while token-based models are more resilient to this attack."

And the abstract: *"Statistical approaches to detection have semantic blind
spots, but semantic approaches are susceptible to persona-prompting."*

**This is independent confirmation of rung 4, on the same corpus, by a different
method** — frontier zero-shot judges fall ~95.8% → ~59.0% under persona
prompting, the same direction and rough magnitude as our A2 (0.866 → 0.49/0.45).

**But one ITB result cuts against our stronger claim, and it matters more than
the confirmation.** GPTZero — a statistical detector trained on generic AI text,
*not* on this corpus — **holds its performance across the persona split.** That is
positive evidence that a **persona-invariant signal exists in this corpus**, which
our estimators failed to learn.

The distinction is not token-vs-semantic, as ITB frames it. It is **trained
in-corpus vs trained out-of-corpus**:

| estimator | trained on | persona robustness |
|---|---|---|
| our A0 (TF-IDF) | **this corpus**, one persona family | collapses (0.960 → 0.60/0.50) **[R]** |
| our A2 (frozen rep + head) | **this corpus**, one persona family | collapses (0.866 → 0.49/0.45) **[R]** |
| ITB's Opus-4.6 / GPT-5.5 | nothing (zero-shot) | collapses (~95.8% → ~59.0%) **[F]** |
| **GPTZero** | **generic AI text, out-of-corpus** | **consistent** **[F]** |

**Consequence for the record.** The 2026-08-18 correction narrowed "the ceiling is
the corpus's" to a two-method statement. That narrowing was right, and GPTZero's
persistence now makes the stronger claim **positively unlikely**, not merely
unproven: the limitation looks like a property of *fitting an estimator to this
corpus's small, persona-imbalanced training set*, not of the corpus's information
content. Recorded as a further correction in `v2/DECISIONS.md`.

**Dual-use caution — verified verbatim**, and it is in the paper, not merely
inferred (registry §10 previously cited it without a quotation):

> "Risks of this study include use of the benchmark or its dataset for training
> LLMs to be less detectable; to adapt the benchmark for such future challenges,
> we would re-release with more complex dialogues."

Authors: William Hager, Ishika Rathi, Masum Hasan, **Cameron Jones** — confirming
the design's note that an author of the source data co-authors the benchmark and
its caution. **[F]**

---

## 2. Active information seeking — the group that constrains Stage C

| source | question | method | tasks | baselines | headline | licence |
|---|---|---|---|---|---|---|
| **Uncertainty of Thoughts** (2402.03271) | can an LLM choose questions that maximise expected information gain? **[A]** | uncertainty-aware simulation + information-gain reward + reward propagation **[A]** | medical diagnosis, troubleshooting, 20-questions **[A]** | direct prompting **[A]** | **+38.1% average success rate** vs direct prompting **[A]** | CC BY 4.0 **[A]** |
| **BED-LLM** (2508.21184, **ICLR 2025**) | adaptive information gathering via sequential Bayesian experimental design **[A]** | iteratively maximise **EIG** under a probabilistic model from the LLM's belief distribution; specialised EIG estimator + candidate-query proposal **[A]** | 20-questions, active user-preference inference **[A]** | direct prompting, other adaptive designs **[A]** | **"typically more than double" baseline; GPT-4o 93% vs prompt-only 45%** on Animals **[A]** | Apple ML Research **[A]** |
| **CA-BED** (2606.01182, ICLR 2026 workshop) | question selection under ambiguous / partially informative answers **[A]** | conversation-aware BED + LLM-based likelihood estimation, inference-time planning **[A]** | two entity-deduction benchmarks **[A]** | direct prompting, other info-seeking methods **[A]** | **+21.8% success, +1.8 turns** vs direct prompting **[A]** | CC BY 4.0 **[A]** |
| **ClarQ-LLM** (2409.06097) | can LLMs ask clarifying questions in task-oriented dialogue? **[A]** | benchmark with an **executable provider agent** replicating the human provider **[A]** | 31 task types × 10 scenarios = **310**, English–Chinese **[A]** | — | **LLAMA3.1-405B max 60.05% success** **[A]** | arXiv nonexclusive-distrib **[A]** |

### 2.1 ⚠️ The field has moved past UoT, and Stage C's plan has not

`v2/experiment_design.md` §13.2 names **one** active baseline — "a UoT-style
expected-information-gain policy" (arm B2). The matrix now shows that is
out of date:

- **BED-LLM (ICLR 2025)** is the same idea done more carefully — an explicit
  EIG estimator over a posterior derived from the model's belief distribution,
  rather than UoT's simulation-and-propagation heuristic — and reports far larger
  gains (GPT-4o 93% vs 45% prompt-only on 20-questions-style Animals).
- **CA-BED (2026)** extends it to exactly the case our respondents produce:
  **ambiguous or partially informative answers.** That is the D0 setting.
- **ClarQ-LLM** supplies something no real corpus in our source-coverage table
  does: an **executable provider agent**, i.e. requirement **R4**. It is a
  candidate Stage C environment, or at least a design precedent for D0's renderer.

**Recommendation for Stage C, for the owner to decide before D0 is built:**
arm B2 should be **BED-LLM-style EIG**, with UoT retained as a secondary
comparison rather than the primary named baseline. Gate 2A asks whether a
pre-registered non-trained active policy beats random and fixed; picking the
weaker of two published methods would make that gate easier to pass and the
result less meaningful. **This is a design-doc change and is not made
unilaterally** — recorded in `v2/DECISIONS.md` as a recommendation.

---

## 3. Calibration and confidence training

| source | question | method | metrics | licence |
|---|---|---|---|---|
| **ConfTuner** (2508.18847) | train calibrated *verbal* confidence without ground-truth confidence labels **[A]** | fine-tune on the **tokenized Brier score**, proven a proper scoring rule **[A]** | calibration; downstream self-correction and model-cascade gains **[A]** | CC BY 4.0 **[A]** |
| **LACIE** (2405.21028) | calibrate explicit *and implicit* confidence markers **[A]** | listener-aware preference optimisation; speaker/listener two-agent game **[A]** | confidence separation, human acceptance, TruthfulQA generalisation **[A]** | CC BY 4.0 **[A]** |

Directly live: P1 was re-anchored to calibration, and A2 was found to have **no**
calibrator. If one is added, ConfTuner's tokenized Brier is the named method in
§13.3 and is now verified as a proper scoring rule.

---

## 4. CoT faithfulness and monitoring (frames §7.2; nothing measured depends on it)

| source | finding as used here | licence |
|---|---|---|
| Measuring Faithfulness in CoT (2307.13702) | reliance on CoT is task-dependent, and **larger models produce less faithful reasoning** **[A]** | arXiv nonexclusive **[A]** |
| LMs Don't Always Say What They Think (2305.04388) | CoT is swayed by biasing features the model **omits from its stated reasoning** **[A]** | CC BY 4.0 **[A]** |
| Monitoring Reasoning Models (2503.11926) | CoT monitors help, but optimisation pressure produces **obfuscated reward hacking** **[A]** | arXiv nonexclusive **[A]** |
| CoT Monitorability (2507.11473) | monitorability is a real but **fragile** safety opportunity; 41-author multi-org **[A]** | CC BY 4.0 **[A]** |
| Emergent Misalignment (2502.17424) | narrow finetuning (insecure code) induced **broad** misalignment **[A]** | arXiv nonexclusive **[A]** |
| LIMA (2305.11206) | **1,000 curated examples** suffice for alignment-style SFT **[A]** | CC BY 4.0 **[A]** |
| Sleeper Agents (2401.05566) | **[U] not fetched** | **[U]** |

---

## Remaining gaps

| item | why it is still open | priority |
|---|---|---|
| ITB dataset **registry record** (§8.2) | the benchmark CSV is a separate source; its terms are unregistered, and it is now cited for verified numbers | **high** — cheap, and it is a Gate 0 requirement |
| Sleeper Agents (2401.05566) | framing only | low |
| SPADE, HC3, HANSEN | dataset candidates, deprioritised once Track A concluded | low unless a real-passive replication is revived |
| Full-text pass on Group 2 | all four are **[A]**, and B2's choice now rests on them | **high before D0 is built** |

**Bottom line:** the matrix is complete enough to make Stage C decisions, with
one caveat — the Group 2 recommendation rests on abstracts, so a full-text pass on
BED-LLM and CA-BED should precede committing arm B2.
