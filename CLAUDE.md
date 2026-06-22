# Sherlock Investigates — Claude Context

Fine-tuning and adversarial conversation experiment at the intersection of LLM fine-tuning, deception detection, and chain-of-thought analysis.

## Workspace
Part of the Post-Uni Projects workspace. See `../PROJECTS.md` for the full project index.

---

## What this is

**Phase 1** — Fine-tune small open-weights models on different corpora to produce variants with distinguishable reasoning priors.

**Phase 2** — Place pairs of variants into adversarial conversations where each agent tries to identify the other as human or AI while passing as human itself. The novel measurement is the **temporal gap** between when an agent first becomes suspicious in its private chain-of-thought and when it commits to a decision in its visible utterances.

Full design rationale and all decisions: `EXPERIMENT_DESIGN.md`.

---

## Stack

- Python (venv — `python3 -m venv venv && source venv/bin/activate`)
- QLoRA fine-tuning: 4-bit NF4 quantization, rank 32 / alpha 64
- Base models: Qwen2.5-7B-Instruct, Mistral-7B-v0.3
- Training corpus: Sherlock Holmes canon (~60K words + 3-5x augmentation)
- Held-out: *The Adventure of the Speckled Band*
- Compute: RunPod Community RTX 4090 (~$0.34/hr), total budget $5–8

---

## Repo layout

```
data/
  raw/          Gutenberg downloads, untouched
  processed/    Stripped and normalised text, train/heldout splits
  augmented/    Reformatted training data (after augmentation pipeline)
  probes/       Behavioural probe prompt sets (tracked in git)
scripts/
  data_prep/    Extraction and augmentation pipeline
  training/     LoRA fine-tuning scripts
  eval/         Perplexity and behavioural probe scripts
  conversation/ Conversation orchestration
  analysis/     Statistical analysis notebooks
configs/        YAML hyperparameter configs per run
results/
  pilot/        Pilot perplexity, probes, generation samples
  full/         Full experiment data
  analysis/     Final figures, tables, statistical outputs
```

---

## Quick start

```bash
make install   # venv + pinned requirements
make help      # list all targets
make run       # full data pipeline: download → chunk → classify → augment
make test      # smoke tests (pure logic — no network, no Ollama)
```

The Makefile is the canonical entry point; it calls `venv/bin/python`
directly, so no activation needed. `classify`/`augment` are served from
on-disk caches on re-runs and only need Ollama on cache misses. Training is
GPU-only — see `docs/runpod-runbook.md`.

---

## Key conventions

- All scripts are run from the repo root with the venv active.
- `configs/` is the single source of truth for hyperparameters — don't hardcode values in scripts.
- The full data corpus (raw, processed, augmented, probes) is tracked in git — it's small (~8MB) and irreplaceable. Model weights and LLM-response caches are gitignored.
- `results/pilot/` is append-only — never overwrite; use timestamped filenames.

---

## Decision Log

**Every architectural or experimental design decision must be logged here.** Append an entry whenever a non-trivial choice is made — model selection, schema design, statistical methodology, corpus scope, infrastructure path, hypothesis framing. Never edit or delete existing entries; the log is append-only and read chronologically.

Entry format:
```
### YYYY-MM-DD — [Short title]
**Decision:** What was decided.
**Reasoning:** Why this was chosen over alternatives, including the specific evidence or constraint that drove the choice.
**Alternatives considered:** What was ruled out and why.
```

---

### 2026-06-16 — Inference schema: 1-structured-call over 3-call (think/speak/probe)

**Decision:** Per-turn agent output uses a single LLM call returning structured JSON `{reply, suspicion_score, reasoning_trace, cues[], trap_strategy, public_accusation}`, enforced via vLLM `guided_json`, rather than three separate calls (think → speak → probe).

**Reasoning:** Kaggle T4 sessions are 9-hour capped. At ~5 seconds/generation, the 3-call approach yields ~90 conversations per session vs ~270 for the 1-call approach. Target conversation volume is ~1000+ runs across the pairing matrix. The 3× efficiency gain is necessary to complete the experiment within free-tier budget. The `reasoning_trace` field in the schema captures what the think call would have captured; `suspicion_score` replaces the separate probe call. The trap_strategy field adds forward-planning visibility not present in the original 3-call design.

**Alternatives considered:** 3-call (think/speak/probe) approach — cleaner separation between reasoning and utterance generation, but 3× slower and would require ~67 Kaggle sessions vs ~22. Mac-local Ollama for conversations — feasible at 20–35 tok/s but 30–50 hours wall-clock for 1000 conversations (several days of background running). Ruled out both in favour of Kaggle + vLLM.

---

### 2026-06-16 — Hypothesis framing: confirmatory vs exploratory split

**Decision:** Pilot evaluation gates (perplexity shift ≥5%, WikiText flat, MMLU <3pp drop, behavioral probe separation on deduction-inviting prompts) are **confirmatory** hypotheses supported by prior work. The commitment gap direction (faster vs slower), non-monotonic suspicion curve frequency, and private/public divergence patterns are **exploratory** — stated with theoretical motivation but no directional pre-commitment.

**Reasoning:** Confirmatory claims require prior empirical evidence for the stated direction; the perplexity and probe gates have direct support from AuthorMix (style transfer via LoRA) and Betley et al. (dose-response curve). The commitment gap has no prior empirical precedent — two competing theoretical accounts (confident-deducer → commits faster; systematic-reasoner → gathers more evidence → commits later) have equal face validity. Claiming a direction without evidence would be vibes-based and unjustifiable in a research note.

**Alternatives considered:** Pre-committing a directional hypothesis for the commitment gap based on the Holmes-as-confident-deducer narrative — rejected because no empirical work supports either direction and the claim would be falsified by the opposing mechanism, which is equally plausible.

---

### 2026-06-16 — Build order: orchestrator before full canon pipeline

**Decision:** Build and validate the conversation orchestrator against base models before running the full Sherlock canon through the classify → augment pipeline.

**Reasoning:** The augmentation framings (Q&A, reverse-construction, Watson-summary) should align with how the model is prompted at inference time. The orchestrator defines those inference-time prompts. Discovering a prompt format mismatch after running the full pipeline (hours of Ollama + Claude API calls on ~8000+ chunks) would require re-augmentation. Validating the orchestrator first against base Qwen2.5-7B via Ollama costs nothing and surface schema/prompt issues before any GPU spend.

**Alternatives considered:** Run full canon pipeline in parallel with orchestrator development — viable only if training objective is confirmed to be purely causal LM on text (which it is), meaning the training format and inference format are decoupled. Partially valid, but the augmentation framing choices could still be informed by what conversation prompts work well, so orchestrator-first is still preferred.

---

### 2026-06-17 — Prompt engineering: per-turn JSON reminder + response_format enforcement

**Decision:** Two prompt-engineering fixes applied after pilot validation revealed 84% fallback-parse rate with base Qwen2.5-7B: (1) append `_JSON_REMINDER` to every user-side message in `_build_messages`; (2) add `response_format={"type": "json_object"}` alongside `guided_json` in the API call.

**Reasoning:** qwen2.5:7b stays in JSON mode for the first 3–4 turns then reverts to chat mode as context grows and the system prompt gets buried. The per-user-message reminder creates a consistent call-and-response reinforcement throughout the context window. The `response_format` parameter is Ollama's native JSON enforcement as belt-and-suspenders. After both fixes: 93% real-parse rate, 97% reasoning_trace/cues population, reply field clean (no leaked analysis).

**Alternatives considered:** Format injection via post-prompt only (original design) — insufficient as context window grows. Full vLLM deployment for strict schema enforcement — not available in local Ollama setup; deferred to Kaggle production runs where vLLM serves the fine-tuned adapters.

---

### 2026-06-17 — Prompt field separation: reply vs reasoning_trace

**Decision:** Tightened `_JSON_BLOCK` descriptions to explicitly separate the two fields: reply = "the words you literally speak to the other person — natural human speech only; put NO analysis or commentary here"; reasoning_trace = "your private detective notes — clues you noticed and what they mean; this never reaches the other party."

**Reasoning:** Without explicit separation, qwen2.5:7b put reasoning analysis in the reply field (e.g. reply = "Simple and natural response, no unexpected knowledge or overly sophisticated language…"). This contaminates the public channel with private reasoning and breaks the core experimental invariant. The fix eliminates the bleed: after applying, reply fields contain only natural spoken text.

**Alternatives considered:** Separate system prompts for reply vs JSON generation (two-stage output) — adds complexity and cost without addressing the root cause (model doesn't distinguish private from public within a single JSON output). Rejected in favour of clearer field descriptions.

---

### 2026-06-16 — Full Holmes canon scope

**Decision:** Download the complete Conan Doyle canon from Project Gutenberg for the full experiment: 4 novels + 5 short-story collections (~600K raw words). Raw files go to `data/raw/canon/`; processed files to `data/processed/full_canon/`. Pilot continues to use only the 3-story subset already in `data/processed/`.

**Reasoning:** Full experiment target is 1.8M–3M effective tokens post-augmentation (full canon × 3-5× augmentation), which clears the 1M-token threshold for reliable behavioral shift. The raw text is small (~4MB total) and free. Downloading now avoids a blocking step later when starting full-experiment training.

**Alternatives considered:** Continue with pilot corpus only until pilot passes evaluation gates — would create a blocking dependency (can't start full pipeline until pilot results are in hand). Raw download has zero cost and no downside to doing it early.

---

### 2026-06-21 — Control corpus redesign: Victorian fiction + legal opinions replace scrambled-Sherlock + medical

**Decision:** Replace the original scrambled-Sherlock and medical-case-reports controls with (1) same-era Victorian/Edwardian fiction (Dickens, Austen, Hardy — Project Gutenberg) and (2) public-domain legal opinions (CourtListener). Both deferred until pilot eval gates pass.

**Reasoning:** Scrambled-Sherlock (sentence-shuffle within passages) is artificial — individual sentences still carry reasoning vocabulary ("therefore," "I observe"), so the "no deductive structure" condition is contaminated. A real corpus with no deductive chains is a cleaner control; Victorian fiction on Gutenberg costs nothing and runs through the existing pipeline unchanged. Medical case reports were rejected because medical reasoning is probabilistic and differential (ruling hypotheses in/out) whereas Holmes's reasoning is declarative and certain — "same deductive structure" is imprecise for that pairing. Legal opinions (facts → statute applied → ruling) are structurally closer to the observation→inference→conclusion pattern in Holmes and available freely via CourtListener. Deferral rationale: building control corpora before the primary manipulation (base vs. Sherlock) is confirmed to produce a measurable effect inverts the experiment's staged logic.

**Alternatives considered:** Scrambled-Sherlock — rejected (artificial confounds, noisy "no structure" condition). Medical case reports — rejected (differential reasoning ≠ declarative deduction; structural mismatch). Sentence-level vs. paragraph-level vs. cross-story shuffling all considered and all rejected for the same reason: any shuffling is artifactual.

---

## Current state (update each session)

**Last updated: 2026-06-19**

### Tooling (2026-06-11 organisation pass)
- Makefile added — canonical targets above, all verified against the committed corpus (cached re-runs reproduce tracked outputs)
- `tests/test_smoke.py` — 12 pure-logic smoke tests via `make test`; also proves `AUGMENT_VERSION`/`PROMPT_VERSION` gate their caches
- CI (`.github/workflows/ci.yml`) — install/lint/test on push and PR, green
- `docs/runpod-runbook.md` — pod spec, mounts, deferred training installs (pre-flight; update after first real pod run)

### Phase 1 — Data prep: COMPLETE (pilot corpus)
- Corpus downloaded and cleaned: A Study in Scarlet + Scandal in Bohemia + Red-Headed League (training); Speckled Band (held-out)
- 957 chunks labeled by qwen2.5:7b → `data/processed/chunks_labeled.jsonl`
- Augmentation pipeline run → `data/augmented/train.jsonl` (1168 examples, 325K tokens, central ×3 oversample)
- Behavioral probe set written → `data/probes/probe_set_v1.jsonl` (30 prompts)
- Both YAML configs fully populated → `configs/pilot_qwen.yaml`, `configs/pilot_mistral.yaml`

### Phase 1 — Data prep: COMPLETE (full canon)
- All 9 works chunked → `data/processed/full_canon_chunks.jsonl` (10,409 chunks)
  - Speckled Band excluded as held-out; case_book/his_last_bow/return had no roman-numeral headings → chunked as novels
- Full canon labeled → `data/processed/full_canon_chunks_labeled.jsonl` (11h21m, 957 pilot cache hits)
  - Distribution: none=7830 (75.2%), minor=1843 (17.7%), central=736 (7.1%), errors=0
- Full canon augmented → `data/augmented/full_canon_train.jsonl` (21h58m, 5448 cache hits)
  - **12,999 examples, 0 errors, ~3.44M tokens** (central ×3 oversample)
  - central: 9858 total (VERBATIM=2208, QA=2208, WATSON=2139, CHAIN=2208, REVERSE=1095)
  - minor: 3141 total (VERBATIM=1843, WATSON=1298)
  - Token note: 3.44M >> pilot 400K target; this is expected for full canon. Well above 1M behavioral-shift threshold.

### Phase 1 — Training: NOT STARTED
- Training script written → `scripts/training/train_lora.py`
- Orchestrator blocker resolved (validated 2026-06-16)
- **Next action:** spin up Kaggle T4, point training config at `data/augmented/full_canon_train.jsonl`, run QLoRA on Qwen2.5-7B

### Phase 2 — Conversation orchestrator: VALIDATED
- Schema: 1-structured-call returning `{reply, suspicion_score, reasoning_trace, cues[], trap_strategy, public_accusation}`
- Files: `schema.py`, `agent.py`, `prompts.py`, `conv_logging.py`, `orchestrator.py`, `run_pilot.py`
- Validated end-to-end: 5 conversations × 12 turns × 2 agents = 120 turn records, JSONL schema correct
- Field population after prompt fixes: suspicion_score real parse 93%, reasoning_trace 97%, cues 97%, trap_strategy non-none 93%
- Pilot outputs in `results/pilot/conversations/turns_20260616_*.jsonl` and `conversations_20260616_*.jsonl`
- **Baseline behavior (base vs base):** scores stay 0.1–0.5, no accusations, no commitment gaps — expected; Holmes variants should push scores higher
- **Next action:** run Kaggle training on both base models; then re-run orchestrator with fine-tuned adapters

### Supporting artifacts
- `data/augmented/augmentation_spec.md` — framing templates + worked examples
- `data/augmented/manifest.json` — full canon augmentation manifest (model, version, counts, per-framing breakdown)
- `results/pilot/chunk_audit_report.md` — label quality audit
- `results/analysis/literature_notes.md` — structured notes on all cited papers
- `results/pilot/pilot_writeup_template.md` — pre-filled writeup template (fill in after eval)

### Key numbers to remember
- Training corpus (pilot): 1168 examples, ~325K tokens
- Training corpus (full canon): 12,999 examples, ~3.44M tokens → `data/augmented/full_canon_train.jsonl`
- Held-out: `data/processed/heldout/speckled_band.txt`
- Perplexity pass gate: ≥5% drop on Speckled Band, WikiText within ±5% of base
- MMLU pass gate: <3pp drop vs base
- Suspicion threshold for "suspicion event": P(AI) ≥ 0.7
- Target conversation volume: ~1000 (pilot pairing matrix subset); ~6000 (full 10×10 matrix)
- Inference compute: Kaggle T4 free tier, 1-structured-call, ~270 conversations/9hr session
