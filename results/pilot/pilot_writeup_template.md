# Sherlock Investigates: Pilot Writeup

*[INSTRUCTIONS FOR FILLING IN THIS TEMPLATE: This document is pre-filled as far as possible from the experiment design. Sections marked with italicized placeholder text describe what to add after training and evaluation complete. Do not modify the pre-filled content unless the pilot deviated from the design. Target length for the completed writeup: 2,000–4,000 words. Suitable for a HuggingFace blog post or personal blog post on Padawan Coder.]*

---

## 1. What This Experiment Is

This post documents the pilot phase of Sherlock Investigates — a personal research project at the intersection of LLM fine-tuning, deception detection, and chain-of-thought analysis.

The experiment has two phases. Phase 1 fine-tunes small open-weights language models on different text corpora to produce variants with distinguishable reasoning priors. Phase 2 places pairs of those variants into adversarial conversations where each agent tries to identify the other as human or AI while passing as human itself. The novel measurement in Phase 2 is the **temporal gap** between when an agent first becomes suspicious in its private chain of thought and when it commits to a decision in its visible utterances.

The Sherlock Holmes corpus is the manipulation material for Phase 1. The hypothesis is that training on Holmes canon — deductive narrative told from Watson's observational perspective — will shift a model's reasoning priors in a measurable direction: more structured inference from observation, more commitment to a specific conclusion, more Victorian register. The experiment is not really about whether the model starts to sound like Holmes. It is about whether models with different reasoning priors show measurably different patterns in their chain-of-thought versus their visible utterances when placed in adversarial scenarios.

This pilot answers two questions before the full experiment runs: (1) can the chosen base-and-config produce a behaviorally distinguishable variant from a corpus of this size and shape, and (2) which of the two candidate base models responds better to the manipulation?

---

## 2. Why the Data Volume Question Matters

The central scientific risk in this experiment is that the training corpus is too small to produce behavioral shift. A 60,000-word Sherlock corpus is roughly 80,000 tokens in raw form. Three independent lines of evidence in the fine-tuning literature converge on a million tokens as the threshold below which behavioral shift on 7B-class models is unreliable (LIMA, Long Is More for Alignment, and Betley et al.'s emergent misalignment work — see `results/analysis/literature_notes.md` for structured entries on each).

This is more than ten times below the documented threshold. The data volume problem is addressed through two mitigations. First, a multi-framing augmentation pipeline reformats each passage into five different framings (raw narrative, Q&A, Watson retrospective, structured chain, reverse construction), pushing the effective training token count from roughly 46,000 tokens (central and minor chunks with oversampling) to an estimated 140,000–230,000 tokens. Second, the pilot is framed as a **calibration measurement** rather than an assumption of success: finding that 80,000–230,000 tokens does not produce behavioral shift is itself a clean empirical finding about the data-volume threshold for reasoning-prior induction on this class of model.

*[After training: Add one sentence stating the actual effective token count of the augmented training corpus as produced by the pipeline. Add one sentence noting whether this was within the 240K–400K target range.]*

---

## 3. Pilot Configuration

All hyperparameters are drawn from `EXPERIMENT_DESIGN.md` and committed to `configs/` before training began. No values were changed during the run.

| Component | Specification |
|---|---|
| Base models | Qwen2.5-7B-Instruct (Unsloth fixed checkpoint), Mistral-7B-v0.3 |
| Adapter method | QLoRA, 4-bit NF4 quantization |
| LoRA rank / alpha | 32 / 64 |
| Target modules | All 7 linear layers: q, k, v, o, gate, up, down |
| Learning rate | 1e-4, cosine schedule, 5% warmup |
| Effective batch size | 16 (via gradient accumulation) |
| Epochs | 3 |
| Sequence length | 2,048, packed within document boundaries |
| Seed | 42 |
| Training corpus | Central + minor chunks, central 3× oversampled, 5 augmentation framings |
| Effective training tokens | *[fill in after augmentation pipeline runs]* |
| Held-out corpus | "The Adventure of the Speckled Band" (~8,000 words) |
| Compute | RunPod Community RTX 4090 (~$0.34/hr) |
| Actual wall-clock time | *[fill in after training]* |
| Actual compute cost | *[fill in after training]* |

---

## 4. Results — Perplexity

Perplexity was evaluated on three corpora for each trained adapter and for the base model: the held-out Sherlock story ("The Adventure of the Speckled Band"), a 250,000-token sample from WikiText-2, and optionally the domain-control corpus.

**Thresholds (pre-committed from design doc):**
- Sherlock held-out: variant must show at least 5% lower perplexity than base to pass
- WikiText-2: variant must stay within 5% of base in either direction
- If WikiText drops more than 5%: over-specialization flag — investigate before proceeding
- If Sherlock held-out drops less than 5%: manipulation too weak — do not scale up

| Model | Sherlock held-out PPL | WikiText-2 PPL | Domain control PPL | Pass/Fail |
|---|---|---|---|---|
| Qwen2.5-7B-Instruct (base) | *[fill in]* | *[fill in]* | *[fill in]* | baseline |
| Qwen2.5-7B-Instruct (Sherlock) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |
| Mistral-7B-v0.3 (base) | *[fill in]* | *[fill in]* | *[fill in]* | baseline |
| Mistral-7B-v0.3 (Sherlock) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |

*[After evaluation: Fill in the table. Add one paragraph interpreting the numbers — does the perplexity pattern look like domain specialization (Sherlock PPL down, WikiText flat) or something else? Note any anomalies.]*

---

## 5. Results — Behavioral Probes

The probe set (`data/probes/probe_set_v1.jsonl`) consists of 30 prompts in three categories of 10: NEUTRAL (everyday conversation), DEDUCTION_INVITING (observational reasoning prompts), and REASONING_REQUIRED (logic puzzles and argument analysis). Each variant generated 3 samples per prompt at temperature 0.7, producing 90 generations per condition. Generations were scored by Claude-as-judge on five dimensions; a 20% sample was hand-validated.

**Scoring dimensions (pre-committed):**
1. Deductive language use (frequency of explicit inference markers)
2. Observational specificity (how many distinct details are named)
3. Response length (token count)
4. Hedging frequency (expressions of uncertainty)
5. Victorian register (lexical and syntactic formality score)

**Expected effect directions (pre-committed):**
- DEDUCTION_INVITING prompts: Sherlock variant should score higher on dimensions 1 and 2; may show higher dimension 5
- NEUTRAL prompts: no significant difference expected
- REASONING_REQUIRED prompts: no significant difference expected (capability check)

| Dimension | Qwen base | Qwen Sherlock | Mistral base | Mistral Sherlock |
|---|---|---|---|---|
| Deductive language (DEDUCTION_INVITING) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |
| Observational specificity (DEDUCTION_INVITING) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |
| Response length (DEDUCTION_INVITING) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |
| Hedging frequency (DEDUCTION_INVITING) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |
| Victorian register (DEDUCTION_INVITING) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |
| Deductive language (NEUTRAL) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |
| Deductive language (REASONING_REQUIRED) | *[fill in]* | *[fill in]* | *[fill in]* | *[fill in]* |

*[After evaluation: Fill in the table. Add one paragraph for each base model describing whether the Sherlock variant differs from the base on DEDUCTION_INVITING prompts in the expected direction. Add a paragraph noting any unexpected differences on NEUTRAL or REASONING_REQUIRED prompts. Note the hand-validation rate on the Claude-as-judge scores and whether any dimension was unreliable.]*

*[Quote two or three generation examples side by side — base model vs. Sherlock variant — on the same DEDUCTION_INVITING prompt. Pick a representative example where the difference is visible, if one exists.]*

---

## 6. Results — MMLU Capability Check

A 100-question MMLU sample across diverse subjects was run on each variant to verify that general capability was not degraded by the fine-tuning.

**Threshold (pre-committed from design doc):**
- Loss of less than 3 percentage points relative to base: acceptable
- Loss of 3–5 percentage points: flag and note in writeup; possibly still usable
- Loss above 5 percentage points: do not proceed — training has damaged the model

| Model | MMLU score | Δ vs. base | Status |
|---|---|---|---|
| Qwen2.5-7B-Instruct (base) | *[fill in]* | baseline | — |
| Qwen2.5-7B-Instruct (Sherlock) | *[fill in]* | *[fill in]* | *[pass/flag/fail]* |
| Mistral-7B-v0.3 (base) | *[fill in]* | baseline | — |
| Mistral-7B-v0.3 (Sherlock) | *[fill in]* | *[fill in]* | *[pass/flag/fail]* |

*[After evaluation: Fill in the table and add one sentence of interpretation. If either model shows a >3pp drop, add a short paragraph noting what this implies for downstream use.]*

---

## 7. What the Numbers Mean

The decision logic from pilot results to next action (reproduced from `EXPERIMENT_DESIGN.md`):

```
                    ┌──────────────────────────────────┐
                    │   PILOT RESULTS                  │
                    │   (perplexity + probes + MMLU)   │
                    └─────────────┬────────────────────┘
                                  │
                  ┌───────────────┴────────────────┐
                  │                                │
        Perplexity shift on             Perplexity shift on
        Sherlock held-out               Sherlock held-out
        ≥ 5%, WikiText flat,            < 5% OR WikiText
        MMLU loss < 3pp                 dropped > 5pp
                  │                                │
                  ▼                                ▼
        ┌──────────────────┐          ┌──────────────────┐
        │  Behavioral      │          │  Manipulation    │
        │  probe gate      │          │  too weak / too  │
        └────────┬─────────┘          │  destructive     │
                 │                    └────────┬─────────┘
        ┌────────┴────────┐                    │
        │                 │              ┌─────┴──────┐
   Probes show        Probes show        │            │
   variant ≠ base     variant ≈ base     ▼            ▼
   on deduction       on deduction   Increase     Reduce LR /
   prompts            prompts        data         epochs /
        │                 │          volume       reduce rank
        ▼                 ▼          via more     and re-pilot
   ┌──────────┐    ┌──────────────┐  augmentation
   │ PROCEED  │    │ Manipulation │  and/or full
   │ to full  │    │ produced     │  Sherlock
   │ experi-  │    │ token-level  │  corpus
   │ ment     │    │ shift but    │
   │ (Base    │    │ not behav-   │
   │ selected │    │ ioral. Try   │
   │ by best  │    │ stronger     │
   │ effect)  │    │ augmentation │
   └──────────┘    └──────────────┘
```

*[After evaluation: Add one paragraph placing the actual results in this flowchart. State clearly which leaf node the results fall into and what the corresponding next step is. Be direct — write "the pilot passes all gates and the next step is X" or "the perplexity gate fails and the next step is Y."]*

---

## 8. Next Step

*This section can be substantially pre-filled because each branch leads to a defined action.*

**If perplexity gate passes and behavioral probes show variant ≠ base:**
Proceed to the full experiment. The base model for the full experiment is selected by which variant shows the largest behavioral probe separation from base on DEDUCTION_INVITING prompts. If both pass equally, Mistral-7B-v0.3 base is preferred (cleaner experimental design with controlled instruction-mix). Full experiment adds domain-control and scrambled-Sherlock conditions, both seeds (42 and 1337), and the conversation phase.

**If perplexity gate passes but behavioral probes show variant ≈ base:**
The manipulation has produced token-level shift but no behavioral change. Options: (a) increase augmentation factor from 5× to 7–8× and re-run on the same base; (b) expand the training corpus to the full Sherlock canon (~600K words, full experiment scale) to push past the 1M token threshold; (c) treat the current pilot as a publishable negative result establishing the minimum data volume for behavioral persona induction in this configuration.

**If perplexity gate fails (shift < 5% on Sherlock held-out):**
The manipulation is too weak. Reduce learning rate or number of epochs (to check for overfitting), increase data volume, or consider switching to a non-instruct base (Qwen2.5-7B base, Mistral-7B-v0.3 base without instruct fine-tuning). Re-pilot before scaling.

**If MMLU drops > 5pp:**
Training has damaged the model. Reduce learning rate, number of epochs, or both, and re-run before proceeding. Do not use a model with >5pp MMLU degradation downstream.

*[After evaluation: Replace the appropriate branch above with the actual decision and its rationale. Keep the other branches for context.]*

---

## 9. Limitations and What This Pilot Cannot Tell Us

The pilot is designed to answer a narrow feasibility question, not to demonstrate that the full experiment will work. Several important things this pilot cannot tell us:

**On data volume:** The pilot tests a single point on the data-volume curve (roughly 140K–230K augmented tokens). It cannot tell us whether 500K or 1M tokens would produce a stronger or more reliable effect. A negative result here is evidence that this volume is insufficient, not that the approach is fundamentally broken.

**On base model generalization:** The pilot trains on two candidate base models but with one seed each. Training stochasticity (seed variance) is not measured. A positive result at seed 42 might not replicate at seed 1337, and we will not know this until the full experiment runs. The pilot result is a feasibility signal, not a statistical claim.

**On causal attribution:** If the Sherlock variant differs from the base on behavioral probes, the pilot cannot cleanly decompose whether the effect comes from the deductive reasoning content, the Victorian register, the narrative structure, or some combination. That decomposition is the purpose of the factorial design in the full experiment (Sherlock canon vs. domain-control vs. scrambled-Sherlock vs. base).

**On conversation behavior:** Probe performance does not predict conversation performance. A model that scores higher on DEDUCTION_INVITING probes may or may not show different detection behavior in the twelve-turn adversarial conversation setting. The probes are a necessary but not sufficient gate.

**On CoT faithfulness:** Even if the full experiment shows a gap between private chain-of-thought suspicion and public utterance commitment, the pilot provides no evidence on whether that gap is meaningful or performative. This is the open question the experiment is designed to probe, not one the pilot can answer.

*[After evaluation: Add a sentence or two noting any limitations that emerged from actually running the pilot that were not anticipated in the design — unexpected behavior, evaluation quirks, compute surprises, etc.]*
