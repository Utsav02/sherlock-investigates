# Cowork Task Guide — Sherlock Investigates

Claude Cowork is used for knowledge-work tasks: reading local files, synthesizing
information, and producing polished deliverables. Code iteration and debugging
stay in Claude Code.

---

## Folder access to grant

| Folder / file | Why |
|---|---|
| `EXPERIMENT_DESIGN.md` | Primary design reference; every task needs context from it |
| `data/processed/` | Contains labeled chunks (`chunks_labeled.jsonl`), training and heldout story splits |
| `data/augmented/` | Cowork will write augmentation spec and example files here |
| `data/probes/` | Cowork will write the behavioral probe set here |
| `results/pilot/` | Contains `local_model_comparison.json`; Cowork writes audit reports here |
| `results/analysis/` | Cowork writes literature notes and analysis documents here |
| `configs/` | Cowork writes YAML hyperparameter configs here |

Grant **read** access to all of the above. Grant **write** access only to
`data/augmented/`, `data/probes/`, `results/`, and `configs/` — not to
`data/processed/` (those are source files that should not be modified).

---

## Tasks — priority order

Run these in the order listed. Each task produces an artifact that feeds into a
later task or into Claude Code work.

---

### Task 1 (run first) — Chunk label audit report

**Why first**: The labeled chunks in `chunks_labeled.jsonl` were classified by a
local qwen2.5:7b model. Before the augmentation pipeline runs, we need to know
whether those labels are reliable enough to build training data on. The audit
also surfaces how many usable "central" chunks exist, which determines whether
the augmentation target (240K–400K tokens) is achievable.

**Goal prompt** (paste as-is into Cowork):

```
Before we begin, read back my ask and ask any clarifying questions you have.

CONTEXT
I am running a fine-tuning experiment (Sherlock Investigates) where I need to
train a LoRA adapter on Sherlock Holmes passages that emphasize deductive
reasoning. A local qwen2.5:7b model labeled 957 passage chunks from three
stories as "none", "minor", or "central" (deductive content). A separate 20-
chunk comparison in results/pilot/local_model_comparison.json tested qwen2.5:7b
against phi4-mini on the same prompts.

TASK
Read data/processed/chunks_labeled.jsonl and
results/pilot/local_model_comparison.json. Produce an audit report at
results/pilot/chunk_audit_report.md covering:

1. Label distribution table: counts and percentages by label and by source story.
2. Model agreement analysis: on the 20 overlap chunks, how often do phi4-mini and
   qwen2.5:7b agree? Categorize disagreements by direction (phi4-mini labels
   higher vs lower).
3. Quality spot-check: select 5 "central" and 5 "minor" labeled chunks and quote
   each chunk verbatim with the label and justification. For each, give a one-
   sentence assessment of whether the label looks plausible.
4. Usable training data estimate: given that "central" chunks will be 3x
   oversampled and "minor" 1x, estimate the raw token count available before
   augmentation and after 3–5× augmentation. Compare against the 240K–400K
   target from EXPERIMENT_DESIGN.md.
5. Recommendation: one paragraph stating whether the label quality is sufficient
   to proceed with the augmentation pipeline, or whether a Claude second-pass
   re-labeling is needed first.

FORMAT
Markdown document, 600–900 words, with numbered sections matching the five
points above. Use tables for the distribution data.
```

---

### Task 2 — Behavioral probe set

**Why second**: The probe set (30 prompts used to evaluate trained variants) is
needed before training completes. It can be built in parallel with the
augmentation pipeline. Writing it now, while the design is fresh, means it can
be reviewed and refined before the adapters are ready.

**Goal prompt**:

```
Before we begin, read back my ask and ask any clarifying questions you have.

CONTEXT
I am evaluating fine-tuned language model adapters for a research experiment
(Sherlock Investigates). After training, each adapter needs to be tested on a
set of 30 prompts to measure whether it behaves differently from the base model
in ways consistent with Sherlock Holmes deductive reasoning. The full evaluation
methodology is in EXPERIMENT_DESIGN.md (section "The second gate is the
behavioral probe").

The 30 prompts are split into three categories of 10:
- NEUTRAL: everyday small-talk questions (no deductive content expected)
- DEDUCTION_INVITING: prompts that invite observational or deductive reasoning
  (this is where the Sherlock variant should differ from the base)
- REASONING_REQUIRED: logic puzzles and argument-analysis tasks (tests general
  capability, should NOT differ much between variants)

TASK
Read EXPERIMENT_DESIGN.md for the full context. Then generate the 30-prompt
probe set and write it to data/probes/probe_set_v1.jsonl.

Each line of the JSONL must be a single JSON object with fields:
  "id"          integer 0–29
  "category"    one of: NEUTRAL, DEDUCTION_INVITING, REASONING_REQUIRED
  "prompt"      the full prompt text as it will be sent to the model
  "expected_direction"  one sentence describing what the Sherlock variant
                        should do differently (or "no expected difference" for
                        NEUTRAL and REASONING_REQUIRED)

Design notes for DEDUCTION_INVITING prompts: draw on the kinds of observational
details Holmes uses in the training stories (professions inferred from hands/
clothing, travel inferred from posture/luggage, emotional state inferred from
expression). Make prompts feel natural, not test-like. Do not name Holmes or
reference the stories directly — the prompts must work on a model that has no
memory of training.

Also write a companion file data/probes/probe_set_v1_rationale.md (200–300
words) explaining the design choices: why each category has 10 prompts, what
makes a good deduction-inviting prompt, and any specific prompts you are least
confident about.
```

---

### Task 3 — Augmentation framings spec with worked examples

**Why third**: The augmentation pipeline needs concrete prompt templates before
code can be written. Cowork reads the design doc and the actual labeled chunks to
produce templates grounded in the real data, not just abstract descriptions.

**Goal prompt**:

```
Before we begin, read back my ask and ask any clarifying questions you have.

CONTEXT
I am building a training dataset for a fine-tuning experiment. The raw material
is 228 labeled Sherlock Holmes passage chunks (64 "central" + 164 "minor") from
data/processed/chunks_labeled.jsonl. The goal is to augment these 3–5× by
reformatting the same content into different framings, pushing the effective
training token count from ~80K raw toward the 240K–400K target. The five
planned framings are described in EXPERIMENT_DESIGN.md (section "The
augmentation step").

TASK
Read EXPERIMENT_DESIGN.md and data/processed/chunks_labeled.jsonl.

1. Finalize the five augmentation framings. For each, write:
   - Name and one-sentence description
   - The exact system prompt / instruction template a script will use to generate
     that framing from a raw chunk (write these as reusable templates with a
     {{CHUNK}} placeholder)
   - When to apply: should this framing be applied to "central" only, or both
     "central" and "minor"?

2. Select 3 real "central" labeled chunks from the JSONL and produce all five
   augmented versions of each chunk using your templates. Show input and output
   side by side.

3. Write a one-paragraph note on any framings that seem risky (might produce
   low-quality outputs or distort the deductive content) and suggest a fallback.

Write output to data/augmented/augmentation_spec.md. Length: 800–1200 words
plus the 15 worked examples.
```

---

### Task 4 — Pilot hyperparameter config YAMLs

**Why fourth**: Once data prep is done and training is about to start on RunPod,
the configs must exist. Generating them from the design doc now means there is
nothing to look up during a live GPU session.

**Goal prompt**:

```
Before we begin, read back my ask and ask any clarifying questions you have.

CONTEXT
I am about to run fine-tuning training on RunPod. All hyperparameters for the
pilot are specified in EXPERIMENT_DESIGN.md (the table "For quick reference
during execution" and the sections "The fine-tuning architecture, in detail"
and "Training objective and adapter strategy"). The training uses the PEFT /
Unsloth / transformers stack.

TASK
Read EXPERIMENT_DESIGN.md and produce two YAML config files:

configs/pilot_qwen.yaml   — for Qwen2.5-7B-Instruct base
configs/pilot_mistral.yaml — for Mistral-7B-v0.3 base

Each file must include these top-level keys (with appropriate values from the
design doc):
  base_model, adapter_method, quantization, lora_rank, lora_alpha, lora_dropout,
  target_modules (list), learning_rate, lr_scheduler, warmup_ratio,
  per_device_train_batch_size, gradient_accumulation_steps, num_epochs,
  max_seq_length, packing, seed, training_corpus_path, heldout_corpus_path,
  output_dir, logging_steps, save_steps, run_name

Where the two configs differ (base model path, tokenizer quirks, output dir,
run_name), use model-specific values. Where they are identical, use the same
value.

Add a brief YAML comment (# ...) on any line where there is a model-specific
gotcha flagged in EXPERIMENT_DESIGN.md (Qwen pad token, Mistral tokenizer
version, etc.).

Do not invent values not specified in the design doc. For any parameter the
design doc leaves unspecified, write the value as null with a comment:
# not specified in design doc — set before training.
```

---

### Task 5 — Literature synthesis reference note

**Why fifth**: The design doc references ~8 papers by finding (LIMA, "Long Is
More", emergent misalignment, LoRA Land, "LoRA Learns Less", AuthorMix,
QLoRA, and the sequential sampling literature). Having these synthesized in one
place makes the pilot writeup easier and helps catch any outdated assumptions
before training.

**Goal prompt**:

```
Before we begin, read back my ask and ask any clarifying questions you have.

CONTEXT
I am about to run a fine-tuning experiment (Sherlock Investigates). The
experimental design document EXPERIMENT_DESIGN.md cites roughly 8 papers to
justify key design decisions. I need a structured reference note that extracts
each paper's core claim, the specific number or result I am relying on, and
whether any of those numbers might be contested or context-dependent.

TASK
Read EXPERIMENT_DESIGN.md. Identify every paper or empirical finding cited in
the document. For each, produce a structured entry with:
  - Short citation label (e.g. "LIMA (2023)")
  - The specific claim from the paper that the design relies on
  - The exact number or threshold cited (e.g. "~1M tokens for behavioral shift")
  - The section of EXPERIMENT_DESIGN.md that relies on this finding
  - A one-sentence note on potential limitations or caveats (model size, task
    type, domain mismatch relative to this project)

Also write a short synthesis paragraph (150–200 words) identifying which two or
three findings have the most bearing on whether the pilot will succeed or fail.

Write output to results/analysis/literature_notes.md. Aim for a complete entry
for every cited paper, in the order they appear in the design doc.
```

---

### Task 6 — Pilot writeup template

**Why last**: A writeup template is most useful just before the pilot runs, not
before data prep. Run this after Tasks 1–4 are done and training is imminent.

**Goal prompt**:

```
Before we begin, read back my ask and ask any clarifying questions you have.

CONTEXT
I am running a machine learning pilot (Sherlock Investigates) and need to write
a 2,000–4,000 word pilot writeup regardless of outcome — a pre-committed
deliverable described in EXPERIMENT_DESIGN.md (section "Deliverables and
timeline"). The writeup should work as a HuggingFace blog post or personal blog
post. It needs to be honest about what was attempted, what the numbers showed,
and what the next step is.

TASK
Read EXPERIMENT_DESIGN.md in full. Then draft a pilot writeup template in
Markdown at results/pilot/pilot_writeup_template.md.

The template should have these sections, each with a 2–3 sentence placeholder
description (in italics) of what to fill in, followed by any fixed content
that can already be written from the design doc:

1. What this experiment is (can be substantially written now from the design doc)
2. Why the data volume question matters (can be substantially written now)
3. Pilot configuration (populate the table fully from the design doc now)
4. Results — perplexity (placeholder table structure, thresholds pre-filled)
5. Results — behavioral probes (placeholder, scoring dimensions pre-filled)
6. Results — MMLU capability check (placeholder, threshold pre-filled)
7. What the numbers mean (decision flowchart from the design doc, reproduced)
8. Next step (branch on pass / partial pass / fail — can be written now)
9. Limitations and what this pilot cannot tell us

Pre-fill every section as far as possible from the design doc so that filling
in the actual results after training is the only remaining work.
```

---

## Notes on running Cowork tasks

- Always grant Cowork read access to `EXPERIMENT_DESIGN.md` before any task —
  it is the primary context document.
- Run tasks one at a time; each produces an artifact the next can reference.
- If Cowork asks a clarifying question about a number or threshold, the answer
  is almost always in `EXPERIMENT_DESIGN.md` — tell it to re-read the relevant
  section.
- Do not ask Cowork to run Python, install packages, or iterate on code.
  Those tasks stay in Claude Code.
