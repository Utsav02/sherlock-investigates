# Sherlock Investigates — Claude Context

Fine-tuning and adversarial conversation experiment at the intersection of LLM fine-tuning, deception detection, and chain-of-thought analysis.

> **House style.** The "Agent behavior" section at the bottom is shared across all of
> Utsav's personal repos — keep it consistent. Project-specific knowledge lives above it;
> repo-specific overrides are called out explicitly at the point of override.

## Workspace
Part of the Post-Uni Projects workspace. See `../PROJECTS.md` for the full project index.

## Data & secrets — handle with care
- `.env` (gitignored) holds RunPod, HuggingFace, Modal, and any inference-endpoint
  tokens. Never committed; only `.env.example` placeholders should ever be in git.
  Verify nothing secret is staged before every commit.
- **The repo is public by owner decision (settled 2026-08-07).** The corpus, design
  docs, and Decision Log are the load-bearing artifacts of the work — treat them with
  care — but they are published methods, not protected IP: the plan is an open
  methods/negative-result writeup, and nothing here needs to be withheld for novelty.
  Adapters default to private on HF for cost/cleanliness, not secrecy. (Earlier
  language calling these "the experiment's IP" predates the visibility decision.)
- **Model adapters / weights** live on HuggingFace (and optionally locally during
  training). Local weight dirs are gitignored — the git remote does NOT back them up.
  HuggingFace is the authoritative copy; if you train on RunPod, upload before tearing
  the pod down.
- **Modal / RunPod compute is metered.** Total experiment budget is capped at $50 —
  spinning up GPUs is a money decision, not a code decision. Don't deploy or scale up
  without explicit in-session confirmation.

---

## What this is

**Phase 1** — Fine-tune small open-weights models on different corpora to produce variants with distinguishable reasoning priors.

**Phase 2** — Place pairs of variants into adversarial conversations where each agent tries to identify the other as human or AI while passing as human itself. The novel measurement is the **temporal gap** between when an agent first becomes suspicious in its private chain-of-thought and when it commits to a decision in its visible utterances.

Full design rationale and all decisions: `EXPERIMENT_DESIGN.md`.

---

## Related workspace project: decision-traceability

`../decision-traceability/` (added 2026-07-09) is the sibling experiment: same thesis
— when does a model's **visible decision dissociate from its internal state** — on a
toy routing task, measured mechanistically (LoRA-ensemble posterior + per-layer linear
probes) instead of behaviorally. Planned cross-pollination, in priority order:

1. **Fourth commitment level.** Its probe machinery (activation hooks, layer sweep,
   base-model control) ports to this repo's conversation models: a suspicion probe on
   the residual stream gives `t_probe ≤ t_think_07 ≤ t_private_07 ≤ t_public`. The
   `messages_input` field logged in every TurnRecord was built for exactly this replay.
2. **Multi-seed adapters double as a posterior.** The planned multiple-seeds-per-
   condition runs are also a LoRA ensemble — member disagreement on `suspicion_score`
   is an epistemic-uncertainty measurement for free (BALD decomposition in
   `../decision-traceability/lora_ensemble_routing.py`).
3. **Instruction ablation for prompts.py.** Its eval harness measures per-rule
   compliance deltas — directly applicable to finding which prompt rules carry the
   93% JSON parse rate.

---

## Stack

- Python (venv — `python3 -m venv venv && source venv/bin/activate`)
- QLoRA fine-tuning: 4-bit NF4 quantization, rank 32 / alpha 64
- Base models: DeepSeek-R1-Distill-Qwen-7B / 14B / 32B (staged; Unsloth checkpoints)
- Training corpus: Sherlock Holmes full canon (~658K words, 12,999 examples, ~3.44M tokens)
- Held-out: *The Adventure of the Speckled Band*
- Inference: Modal vLLM endpoint (`scripts/inference/modal_app.py`), $30 free credits
- Compute: RunPod RTX 4090 / A40 for training; total budget capped at $50

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
  inference/    Modal vLLM deployment (modal_app.py)
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

### 2026-06-24 — Model upgrade: DeepSeek-R1-Distill-Qwen replaces Qwen2.5 + Mistral base models

**Decision:** Replace the original Qwen2.5-7B-Instruct and Mistral-7B-v0.3 base models with the DeepSeek-R1-Distill-Qwen series (7B → 14B → 32B, staged by budget). The same training script, augmentation corpus, and conversation orchestrator are used unchanged; only the `base_model` config key changes per tier. Total experiment budget capped at $50; inference is free via Modal $30 credits.

**Reasoning:** R1-distill models produce explicit `<think>…</think>` blocks before every response. This transforms the commitment gap from a two-level measure (private `suspicion_score` → public `public_accusation`) into a three-level measure: (1) first suspicion keyword in the think block, (2) `suspicion_score ≥ 0.7` in the JSON, (3) `public_accusation = true`. This is the first experiment to measure commitment gap at all three levels simultaneously. The `think_block` field is logged in every `TurnRecord` alongside `messages_input` (the exact prompt sent to the API), enabling post-hoc TransformerLens activation re-runs for a mechanistic interpretability layer. Staged rollout (7B free validation → 14B ~$2 → 32B ~$6) avoids wasting GPU budget on a model that doesn't pass the eval gates.

**Alternatives considered:** Keep Qwen2.5-7B and Mistral-7B-v0.3 (original plan) — no think blocks, so the commitment gap measurement stays behavioral-only; weaker reasoning baseline makes Holmes behavioral shift harder to detect. Qwen3-32B (native thinking mode via `/think` instruction) — newer and capable, but no published interpretability work specific to Qwen3 thinking tokens; R1-distill has Venhoff et al. (2025) as a direct prior. 70B models — too expensive to fine-tune within $50 budget (~$25-30 for training alone).

---

### 2026-06-24 — Inference platform: Modal (free credits) over Kaggle for R1-distill conversations

**Decision:** Use Modal's free $30 GPU credits to run conversation inference for R1-distill models. The `scripts/inference/modal_app.py` serves vLLM behind a Modal web endpoint; the orchestrator points `AgentConfig.endpoint` at the Modal URL. Kaggle remains the platform for fine-tuning validation (free T4 for 7B).

**Reasoning:** Kaggle 2×T4 (32GB combined) can run 14B inference but is session-capped at 9hr and requires notebook UI interaction. Modal exposes a persistent HTTPS endpoint that the orchestrator can call directly, matches the existing OpenAI-compatible API pattern, and costs nothing within the $30 credit window. For 32B inference (requires ~40GB), Modal A100-40GB is the only free-tier option. The Modal deployment is also a template for the eventual RunPod serverless path if credits run out.

**Alternatives considered:** Kaggle 2×T4 for 14B inference — viable but 9hr cap and manual session management adds friction. RunPod serverless — more control but no free tier; deferred as fallback if Modal credits run out. Together.ai / Groq free tier — do not support loading our custom LoRA adapters; ruled out for fine-tuned model inference.

---

### 2026-06-21 — Control corpus redesign: Victorian fiction + legal opinions replace scrambled-Sherlock + medical

**Decision:** Replace the original scrambled-Sherlock and medical-case-reports controls with (1) same-era Victorian/Edwardian fiction (Dickens, Austen, Hardy — Project Gutenberg) and (2) public-domain legal opinions (CourtListener). Both deferred until pilot eval gates pass.

**Reasoning:** Scrambled-Sherlock (sentence-shuffle within passages) is artificial — individual sentences still carry reasoning vocabulary ("therefore," "I observe"), so the "no deductive structure" condition is contaminated. A real corpus with no deductive chains is a cleaner control; Victorian fiction on Gutenberg costs nothing and runs through the existing pipeline unchanged. Medical case reports were rejected because medical reasoning is probabilistic and differential (ruling hypotheses in/out) whereas Holmes's reasoning is declarative and certain — "same deductive structure" is imprecise for that pairing. Legal opinions (facts → statute applied → ruling) are structurally closer to the observation→inference→conclusion pattern in Holmes and available freely via CourtListener. Deferral rationale: building control corpora before the primary manipulation (base vs. Sherlock) is confirmed to produce a measurable effect inverts the experiment's staged logic.

**Alternatives considered:** Scrambled-Sherlock — rejected (artificial confounds, noisy "no structure" condition). Medical case reports — rejected (differential reasoning ≠ declarative deduction; structural mismatch). Sentence-level vs. paragraph-level vs. cross-story shuffling all considered and all rejected for the same reason: any shuffling is artifactual.

---

### 2026-07-16 — Model retention: stay on R1-Distill despite newer 2026 open models

**Decision:** Keep DeepSeek-R1-Distill-Qwen (7B/14B/32B staged) as the experiment's base models. Newer 2026 reasoning models (Qwen3-Thinking 4B–32B, Apr 2026; GLM-5; Phi-4-reasoning) are NOT adopted for the primary experiment. Qwen3-Thinking-8B/14B is designated the post-pilot replication target: if the commitment gap replicates on R1-Distill-14B, one ~$2 replication run there upgrades the finding from "a property of one distillation" to "a property of thinking models."

**Reasoning:** The experiment's contribution is the measurement, not the model, and the model requirements are explicit think blocks + free-GPU trainability + *well-characterized* thinking behavior — not frontier capability. The interpretability-prior rationale from the 2026-06-24 entry has strengthened since: Thought Anchors (Macar, Bogdan et al. 2025, arXiv 2506.19143) ran its sentence-level CoT analyses on R1-Distill-Qwen-14B — the exact planned model — joining Venhoff et al. as a second published methods baseline. Switching costs (think-tag formats, chat templates, JSON-mode parse-rate retuning, eval-gate re-baselining) are weeks of drift, and the project's dominant risk is not-running, not model staleness.

**Alternatives considered:** Qwen3-Thinking as primary — better benchmarks, same lineage, but ~3 months old with no interpretability literature depth; rejected for the primary, adopted as the replication arm. Starting the pilot on both families in parallel — doubles validation surface before any single result exists; rejected.

---

### 2026-07-06 — Measurement fixes: word-bounded suspicion keywords + api_error turn flagging

**Decision:** (1) Think-block suspicion detection (`conv_logging.py`) now matches keywords with word boundaries (compiled `\b(?:...)\b` regex) instead of plain substring containment. (2) API-call failures in `agent.py` no longer return an unmarked neutral turn: the exception is logged, and the turn is tagged `parse_mode="api_error"` (new field on `TurnOutput`/`TurnRecord`; regex-fallback parses are tagged `"fallback"`, clean parses `"json"`). `compute_conversation_metrics` excludes `api_error` turns entirely. Covered by `tests/test_conv_metrics.py`.

**Reasoning:** Substring matching made `"ai"` fire inside "wait"/"said"/"again" and `"bot"` inside "both", so `t_think_07` would trigger on nearly every English think block — collapsing the novel three-level commitment gap to noise. Separately, a Modal cold-start timeout or endpoint misconfiguration produced turns indistinguishable from real ones (fabricated suspicion_score=0.5 silently breaking the sustained-≥0.7 check, empty reply fed to the opponent). Both are measurement-validity bugs that had to land before any pilot data is collected; results generated with the old code would not have been interpretable.

**Alternatives considered:** Keeping substring matching and post-filtering false positives in analysis — rejected: t_think_07 is computed at logging time and the raw trigger term isn't stored, so contamination would be unrecoverable. Retrying failed API calls inside `generate_turn` — deferred: retries change turn timing/latency semantics; explicit flagging keeps the record honest and lets the orchestrator/analysis decide. Excluding `fallback` turns from metrics as well — rejected for now: fallback parses still reflect genuine model output (93% real-parse rate leaves ~7% fallback), but they're now identifiable in the data if that call changes.

---

### 2026-07-26 — Degenerate self-play invalidates the 2026-07-18 shakedown; detection added

**Decision:** The 2026-07-18 local shakedown is **retracted as evidence of a commitment gap**. Conversation `f217671f` ran 12 turns with **one unique reply** ("You're right; treating people well is always important."), and the second run repeated "Hi! I'm treating you as a human." for its whole length. `t_private_07=6` in that record was computed over a conversation with no accumulating evidence — the `suspicion_score` trajectory (0.8, 0.05, 0.5, 0.5, 0.0, 0.8, 0.7, 0.75, 0.85, 0.8, 0.9, 0.5) is sampling noise on constant input, not a dissociation. Four countermeasures land together: (1) `conv_logging.conversation_degeneracy` computes `unique_reply_ratio` and `max_consecutive_repeats`, flagging `is_degenerate` at 3 consecutive identical replies; (2) the orchestrator terminates such conversations with `termination_reason="degenerate"`; (3) repetition penalties (`frequency_penalty=0.3`, `presence_penalty=0.3`, `repetition_penalty=1.1`) are now `AgentConfig` fields sent on every call; (4) `prompts.PERSONAS` + `persona_pair(seed)` give the two agents different small-talk material, breaking the symmetric self-play fixed point. `run_pilot` reports the degeneracy rate and fails a visible gate above 20%.

**Reasoning:** Two identical policies under near-identical system prompts have a strong attractor: mirror the opponent's last utterance. Nothing in the previous instrument detected this, so the failure was recorded in this file as a success. Any gap metric — commitment gap, survival curve, Cox regression — is meaningless on a transcript with no evolving content, and at the planned n≈1000 this would have produced a fully analysed artefact. Personas deliberately carry no deception-relevant content (no detection hints, no reasoning-style instructions) so they cannot confound the suspicion measurement; they only give the conversation somewhere to go. The 3-repeat threshold is deliberately aggressive — excluding a real conversation costs one sample, keeping a degenerate one corrupts the estimate.

**Alternatives considered:** Raising temperature — treats a structural attractor as a sampling problem and widens variance on every other measure. Different models for A and B — confounds the pairing matrix, which is itself an experimental variable. Post-hoc filtering only, no early termination — retains the diagnosis but wastes ~90% of the turns in a collapsed conversation; both are implemented, but terminating early is what makes a 1000-conversation run affordable. Suppressing the metric entirely on degenerate conversations rather than logging the diagnostics — rejected: the degeneracy *rate* is itself a reportable property of a model pairing.

---

### 2026-07-26 — `t_think_07` redefined: directed suspicion, not topic mention

**Decision:** `t_think_07` now fires on the first think-block sentence that **asserts a suspicion about the interlocutor**, replacing "contains any suspicion keyword." Detection is per-sentence and requires a referent for the other party + a stance verb + a suspicion term (`_DIRECTED_SUSPICION_RE`), or an implicit-referent conclusion such as "I think I'm talking to a bot" (`_IMPLICIT_SUSPICION_RE`). Sentences matching `_TASK_FRAMING_RE` — disjunctive phrasing ("an AI or a human"), assignment restatement ("trying to figure out whether"), explicit hedging ("not sure if"), or self-concealment planning ("convince them I'm not a bot") — are vetoed even when the pattern matches. The superseded measure survives as **`t_think_topic`** on `ConversationRecord`, so analysis can quantify how much the redefinition moved the number.

**Reasoning:** The adversarial system prompt *instructs* the model to determine whether the other party is an AI, so every think block on every turn discusses AI. Measured against the 2026-07-18 pilot data, the old detector fired on **14/14 think blocks (100%)** — a literal constant, and a constant cannot have a gap with anything. Level 2 of the three-level ladder, the level that motivated the entire R1-distill model switch (2026-06-24), was unmeasurable as implemented. The directed detector fires on **8/14 (57%)** of the same blocks and correctly rejects the verbatim turn-0 restatement *"Alright, so I'm trying to figure out if this person they're talking to is an AI or another human"* while accepting the genuine turn-1 conclusion *"someone told me they're an AI language model."* Keeping both measures means the redefinition's effect is auditable rather than asserted.

**Alternatives considered:** A local stance classifier over sentences (Option B in `../experiment.md` §4.2) — higher fidelity but puts a second model inside the measurement instrument, adding its own failure modes and version-pinning burden; still the right upgrade if hand-validation shows precision below ~0.8. A residual-stream probe on think tokens — that is `t_probe`, Level 1, a different and later measurement, not a substitute. Dropping Level 2 entirely and shipping a two-level gap — rejected: the three-level measurement is the project's novelty. **Outstanding:** the hand-labelled validation set (`data/probes/think_stance_labels_v1.jsonl`, ~100 sentences) is NOT yet built; precision/recall of this detector are currently unmeasured, and that must land before the pilot.

---

### 2026-07-26 — Generation seeds derived per (conversation, turn, speaker)

**Decision:** `orchestrator.derive_seed(base_seed, turn_idx, speaker_id)` returns `(base_seed * 100_003 + turn_idx * 2 + speaker_bit) mod (2³¹−1)`, replacing `cfg.seed + turn_idx`.

**Reasoning:** `run_pilot` increments the base seed by 1 per conversation while the orchestrator added `turn_idx` to it, so conversation *i* used seeds {base+i … base+i+11} and conversation *i+1* used {base+i+1 … base+i+12} — **11 of 12 seeds shared between adjacent conversations**, and both agents drew from the same sequence. Replicates were therefore not independent, and any variance estimate, confidence interval, or survival curve computed over them understated variance. At the planned n≈1000 this silently corrupts inference rather than visibly breaking. The stride 100 003 is prime and far exceeds the largest within-conversation offset (2·max_turns+1), so no two conversations can collide; verified over 100 conversations × 24 turns × 2 speakers with zero collisions.

**Alternatives considered:** Hashing `(conv_id, speaker, turn)` — robust but makes seeds unreproducible from the CLI arguments alone, and reproducibility from `--seed` is worth more here than hash quality. Leaving seeds unset for true randomness — loses reproducibility entirely, which the append-only results convention depends on.

---

### 2026-07-26 — `parse_failed` mode: the fallback parser must not return prompt text

**Decision:** New `parse_mode="parse_failed"`, and a new `UNUSABLE_PARSE_MODES = {"api_error", "parse_failed"}` in `schema.py`. When `_fallback_parse` recovers a reply that matches the JSON template's own field descriptions (`agent._looks_like_placeholder`), the whole turn is marked `parse_failed` with an empty reply. The orchestrator now appends **nothing** to either history for any unusable turn, and `compute_conversation_metrics` excludes both modes.

**Reasoning:** At turn 1 of *both* 2026-07-18 runs the fallback regex extracted `"<the words you literally speak to the other person — natural human speech only, under 80 words; put NO analysis or commentary here>"` — the schema's placeholder, not model output — and the orchestrator fed it to the opponent as conversational input. That is the likely seed of the degenerate loop in those conversations. A parser that silently returns the prompt is worse than one that fails, because the corruption propagates into every subsequent turn. Separately, the pre-existing `api_error` path appended an *empty* reply to both histories; that is also now suppressed.

**Alternatives considered:** Retrying the generation on placeholder detection — changes turn timing and latency semantics, the same reason retries were rejected on 2026-07-06; explicit flagging keeps the record honest. Keeping the turn but blanking only the reply — leaves a fabricated `suspicion_score` in the metrics, which is exactly the failure the `api_error` flagging fixed.

---

### 2026-07-26 — Modal A10G cannot serve R1-Distill-14B in bf16

**Decision:** The "Modal A10G (free $30 credits) for 7B/14B" line in Key numbers is corrected: **A10G (24 GB) serves 7B only**. 14B inference requires A100-40GB, or 4-bit AWQ/GPTQ weights on A10G *with a bf16 control arm*. The choice between those two is deferred to the point of deployment and must be logged when made.

**Reasoning:** R1-Distill-Qwen-14B in bf16 needs ~28 GB for weights alone before any KV cache; it does not fit in 24 GB. The plan line was unexecutable as written. Quantising is the cheaper resolution but introduces a confound into a study whose dependent variable is fine-grained token-level behaviour (think-block content and per-turn suspicion), so it cannot be adopted silently — hence the paired bf16 control if that path is taken. Cost consequence: the ~1000-conversation arm moves from ~$3 to ~$12–18, making it the dominant line item rather than training, which is the argument for computing the required *n* from a measured baseline instead of assuming 1000.

**Alternatives considered:** Staying on 7B for the whole experiment — cheapest and removes the problem, but 7B is the free *validation* tier and the pilot gates were written for a stronger main model. Kaggle 2×T4 (32 GB combined) — fits 14B, but the 9-hour session cap and manual notebook management were already rejected on 2026-06-24.

---

### 2026-07-27 — Anti-echo instructions fix the collapse; repetition penalties never could

**Decision:** Degeneracy is countered by **instructions**, not sampling parameters: `_CONVERSATION_RULES` in every system prompt (you are not an assistant; never echo or rephrase the other party; always add something new; no agreement loops), plus the anti-echo rule repeated in the per-turn reminder alongside the JSON reminder. The repetition penalties added 2026-07-26 are retained but are **not** the mechanism.

**Reasoning:** `frequency_penalty` / `presence_penalty` act on tokens already emitted inside the current completion, while the text being mirrored lives in the prompt — they are structurally incapable of preventing turn N+1 from copying turn N. The 2026-07-26 shakedown confirmed this: 6/6 degenerate with penalties and personas active. Adding the rules to the system prompt alone took mirroring 65% → 17% and mean length 4.8 → 13.8 turns, but mirroring returned around turn 4 (seed 1003) — the same burial-with-context decay documented for JSON mode on 2026-06-17. Repeating the rule per turn, exactly as `_JSON_REMINDER` does, is what made it hold. Final: mirroring 13%, all conversations reaching the full 24 messages.

**Alternatives considered:** Raising temperature — treats a structural attractor as a sampling problem. Pairing different models for A and B — confounds the pairing matrix, which is an experimental variable. Both remain available if the fine-tuned adapters reintroduce the collapse.

---

### 2026-07-27 — Degeneracy criterion: locked-or-globally-repetitive, not a bare 3-run

**Decision:** `conversation_degeneracy` flags a conversation when it **locks** (≥5 identical replies consecutively) **or** is **globally repetitive** (<50% distinct replies, once ≥6 replies exist). Replaces "any 3 consecutive identical replies". Early termination follows the same rule.

**Reasoning:** The original criterion used an absolute run length, which does not scale with conversation length — a longer, healthier conversation has *more* opportunities to trip it — and it treated a transient stutter as a collapse. In the `_reminder` run every flagged conversation had `max_consecutive_repeats == 3` exactly with unique-reply ratios of 0.77–0.86, and because detection drives termination, those conversations were killed at 13–14 turns before they could reach an accusation: accusations fell 2/6 → 0/6 *while diversity improved*. The detector had become the binding constraint on the data it existed to protect. Both signals are required because two agents alternating two fixed lines never produce a long run yet carry no accumulating evidence. **Note for anyone rescoring old runs:** recomputing the three earlier runs under this rule returns 0/6 for all of them including the visibly-collapsing baseline — those transcripts were truncated at 3–5 turns by the old rule and the ratio test needs ≥6 replies. Rescoring is circular; Gate 2 was read from a fresh run.

**Alternatives considered:** Ratio-only — misses a late lock in a conversation that started diverse. Run-length-only with a higher threshold — misses the alternating-pair case. Separating the detection threshold from the termination threshold — attractive, and still open if terminating at 5 proves too eager once adapters are in play.

---

### 2026-07-27 — Gate 2 passes; censoring is total and suspicion decays

**Decision:** Gate 2 (≥80% non-degenerate) is **met** on base `deepseek-r1:7b`: 0/6 degenerate, all six conversations reaching the full 24 messages, unique-reply ratios 0.62–0.92. Recorded at n=6; to be re-read at n≈20 before being treated as settled. Full writeup: `results/analysis/shakedown_20260727_writeup.md`.

**Reasoning / what the run exposed, none of which is a code change made silently:**
1. **`t_public` is `None` in 6/6.** No conversation publicly committed within 12 turns, so no commitment gap is computable at the public level and the censoring is 100%. This makes survival analysis mandatory rather than preferred, and makes the *censoring rate itself* a headline result: privately suspicious, never publicly committing.
2. **Suspicion declines over a conversation** (agent A means by turn: 0.67, 0.93, 0.89, 0.71, 0.76, 0.63, 0.48, 0.50). This inverts the model behind `t_private_07`, which requires the score to reach 0.7 **and stay there** — a condition that will rarely be met if scores systematically decay, which is exactly what 1/6 firing shows. **The sustained-threshold definition must be revisited before pilot data collection**; a first-crossing or peak-based definition may be correct. Flagged, not changed.
3. **The legacy topic measure fired at turn 0 in 24/24 conversations across all four runs**, versus 4/6, 3/6, 2/6, 4/6 for the directed measure. The saturation finding is now confirmed well beyond the original two conversations.

**Alternatives considered:** Declaring Gate 2 settled at n=6 — rejected, six conversations of one model pairing is a smoke test. Adjusting the `t_private_07` threshold now to make it fire more often — rejected outright: tuning a measurement definition until it produces results is how a finding becomes an artefact. The definition question gets its own entry once decided.

---

### 2026-07-27 — Open risk: clean-JSON parse rate falls to 60% at 24 turns

**Decision:** Logged as an open risk, **not** resolved. Clean-parse rate by turn band in the gate2 run: 83% (turns 0–3) → 38% (turns 20–23); 60% overall, 39% fallback. The 2026-07-06 decision to retain `fallback` turns in metrics is **explicitly reopened**.

**Reasoning:** That decision rested on the stated premise "93% real-parse rate leaves ~7% fallback". At 39% fallback the premise no longer holds — nearly two in five suspicion scores now come from regex extraction rather than schema-valid output, which is a different measurement. The cause is principally the context-growth decay documented on 2026-06-17, exposed because conversations are now 5× longer; the added anti-echo text in the system prompt and per-turn reminder plausibly contributes by competing with the JSON instruction. Mitigating: Ollama ignores `guided_json` while the production path (vLLM on Modal) enforces it, so much of this may not survive the move — but that is a hypothesis and must be measured on vLLM, not assumed. Either the parse rate is restored or fallback turns are excluded and the exclusion rate reported.

**Alternatives considered:** Excluding fallback turns immediately — would drop 39% of the data on a hypothesis about cause; measure on vLLM first. Trimming the anti-echo text to protect JSON compliance — plausible, but it would trade back the fix that made Gate 2 pass; not without measuring which component costs what.

---

### 2026-07-28 — Stage 0 executed: first GPU hours in the project's history

**Decision:** The Kaggle T4 validation run was executed end to end (`configs/kaggle_t4_validation.yaml`, pilot corpus, 30 optimizer steps, 78 min, $0). Adapter pushed to `utsvsngh/sherlock-r1distill-7b-validation` (private) with the config and training log bundled. **Stage 0 PASSES**: the reasoning format survived fine-tuning.

**Reasoning:** Training was healthy — loss 1.4287 → 0.7553 (final 0.9974 on the last cosine step), grad norms decaying monotonically 0.42 → 0.11, LR schedule landing correctly at 3.1e-07. `trainable params: 80,740,352 || 1.0491%`, which is the exact arithmetic for rank 32 across 7 modules on Qwen2.5-7B (28 layers × 2,883,584) — confirming `modules_to_save` did NOT leak in and the embeddings were untouched. The verification initially reported 0/3 and that was a **false FAIL** caused by the extractor bug logged below, not by the model.

**What this run does NOT establish, and must not be read as:** 311,252 unique tokens over 30 steps cannot shift a reasoning prior. The loss curve is not evidence about the hypothesis. The eval gates (perplexity / WikiText / MMLU / probe separation) were NOT run against this adapter and would produce an uninterpretable null if they were — the honest reading of such a null would be "the dose was too small", which is not a finding.

**Alternatives considered:** Running the eval gates on this adapter anyway to "get a number" — rejected; a null from an under-dosed run is worse than no number because it invites the wrong conclusion.

---

### 2026-07-28 — `torch.cuda.is_bf16_supported()` is unreliable; use compute capability

**Decision:** Introduce `train_lora.native_bf16()`, testing `torch.cuda.get_device_capability()[0] >= 8`. All three call sites migrated (two in `train_lora.py`, one in `verify_think_blocks.py`). The `bnb_4bit_compute_dtype` hardcoded to `torch.bfloat16` in `_load_standard` is gone.

**Reasoning:** Caught live on the Kaggle T4. Since torch ~2.4, `is_bf16_supported()` takes `including_emulation=True` by default and returns **True on Turing (SM 7.5)**, which emulates bf16 in software. The run printed `4-bit compute dtype: torch.bfloat16` on hardware with no bf16 tensor cores; the user's own diagnostic confirmed `is_bf16_supported() == True` but `including_emulation=False → False`. Emulated bf16 is slow and numerically unlike the native path. Two of the three sites were in the same file — `BitsAndBytesConfig` and `TrainingArguments` — so fixing only one would have made the quantization dtype and the training dtype disagree, a failure that surfaces hours into a run. **The `TrainingArguments` occurrence was original code, latent since 2026-06, and would have mis-set the dtype on any Turing GPU.**

**Alternatives considered:** `is_bf16_supported(including_emulation=False)` — correct but only exists on newer torch, so it breaks silently on older ones. Compute capability is stable across versions.

---

### 2026-07-28 — Think blocks with a pre-opened tag were being silently dropped

**Decision:** `_extract_think_block` now handles two shapes: balanced `<think>…</think>`, and **pre-opened** (closing tag only, opening tag consumed by the chat template).

**Reasoning:** Verified on hardware — `unsloth/DeepSeek-R1-Distill-Qwen-7B`'s chat template renders as `<｜begin▁of▁sentence｜><｜User｜>…<｜Assistant｜><think>\n`. The opening tag lives in the **prompt**, so completions carry only `</think>`. The old regex required a balanced pair and returned `(None, text)` — a silent null. **The production impact is what makes this serious:** Ollama and vLLM-with-`--reasoning-parser` return reasoning in a separate field, which is why the 2026-07-18 shakedown captured 14/14 blocks and the bug stayed hidden. A raw vLLM deployment with no reasoning parser — a configuration explicitly on the Modal path in the 2026-06-24 entry — would have produced `think_block=None` on every turn of an entire paid run with nothing in the logs explaining why.

**Alternatives considered:** Stripping the tag in the orchestrator instead — pushes the fix to one call site and leaves the extractor wrong for every other caller. Requiring a reasoning parser on all deployments — a deployment constraint standing in for a code fix, and unenforceable.

---

### 2026-07-28 — Measured T4 throughput rules out 3 epochs on Kaggle

**Decision:** Record 157.4 s/optimizer step for 7B QLoRA at seq 2048, effective batch 16, on a Kaggle T4 (`train_runtime: 4722.8` ÷ 30 steps). Full canon is 1,638 blocks → 102 / 204 / 307 steps at 1 / 2 / 3 epochs → **4.5 h / 8.9 h / 13.4 h**. Three epochs exceeds Kaggle's 12-hour session cap and must not be attempted there.

**Reasoning:** A measured rate, not an estimate. It converts "which corpus and how many epochs" from a guess into arithmetic, and it establishes that any full-canon run beyond ~2 epochs is a RunPod job.

---

### 2026-07-28 — Corpus for the free 7B run: full canon × 1 epoch, not pilot × 3

**Decision:** With 27 h of free Kaggle quota available, the properly-dosed 7B run uses **`data/augmented/full_canon_train.jsonl` for 1 epoch** (~4.5 h), superseding the 2026-07-26 decision to pilot on the 325K corpus for this tier. The pilot corpus remains correct for plumbing validation.

**Reasoning:** The threshold literature measures **unique dataset size**, not tokens processed. LIMA (~1M tokens) trained 15 epochs over its 1M; Betley et al. report 2,000–6,000 examples (~1.2M tokens) for reliable emergence. Measured here: pilot × 3 epochs = 933,756 tokens *seen* but only **311,252 unique** — below LIMA's threshold and near Betley's near-zero floor. Full canon × 1 epoch = **3,356,311 unique**, ~2.8× Betley's emergence point. The earlier decision was made under an assumption of scarce compute that the free quota removes. **Stated as an assumption, not a result:** whether re-reading a small corpus substitutes for a larger one is unsettled — repetition helps with diminishing returns and rising overfitting risk (LoRA Land caps at 1–3 epochs), and no clean exchange rate is published. That is precisely why the unique-token axis is the one used here.

**Also corrected:** the corpus is **3,356,311 tokens**, not the 3.44M recorded elsewhere in this file. The measured chars/token ratio is 4.555 (from 151 blocks × 2048 + 2004 trailing), so the `chars/4` heuristic in the preflight overestimates by ~12%.

**Alternatives considered:** Full canon × 2 epochs (8.9 h) — fits, and is the fallback if 1 epoch under-fits; deferred rather than rejected. Full canon × 3 (13.4 h) — impossible on Kaggle. Staying on the pilot corpus — cheapest, but under-doses the run and risks a null that would be misread as a negative result.

---

### 2026-07-28 — Stage 0 confirmed with evidence; the verifier tested the wrong distribution

**Decision:** `verify_think_blocks.py` now tests the **real task shape** — `prompts.INITIATOR_SYSTEM_THINKING` plus a conversational turn — with one open-ended prompt retained as a labelled stress case at double the token budget. It also reports generated-token count and a `truncated` flag on every probe.

**Reasoning:** Stage 0 is confirmed PASSING on hardware. Measured on the fine-tuned adapter:

| probe | tokens generated | truncated | `</think>` | extracted |
|---|---|---|---|---|
| real task (system prompt + opener) | **422** | No | Yes | 1,293 chars |
| open-ended riddle | 837 | No | Yes | 3,410 chars |

The real-task probe returned a complete think block followed by schema-valid JSON with every field populated — the full production path working on the adapter. 422 tokens / 1,293 chars matches the 2026-07-18 shakedown's 1.4–1.8K chars closely, so **~450 tokens is the planning figure for think-block length on the real task**.

The two false FAILs before this were caused by the verifier testing open-ended riddles, which are a different and much higher-variance distribution: the *same* riddle produced 837 tokens on one sample and exceeded 1,200 on another. Testing an unrepresentative prompt and then concluding the model is broken is precisely the error the repo's own "verify the real artifact and the real path" rule exists to prevent, and it was violated here twice. The `truncated` flag is the field that separates "we cut it off" from "the model never closes the block"; without it the only recourse was raising the cap and guessing.

**Incidental confirmation:** the real-task think block concluded *"I'm leaning towards the user being a human"*, and the directed detector correctly did NOT fire — the negation veto added earlier the same day behaving correctly on live data it was not tuned against.

**Alternatives considered:** Keeping riddle prompts and simply raising the budget — treats a distribution mismatch as a budget problem, and the variance would keep producing intermittent failures. Dropping the open-ended prompt entirely — it is a useful stress case for think-length, so it is retained and labelled rather than removed.

---

### 2026-07-28 — probe_eval.py was scoring the think block, not the answer

**Decision:** `generate_response` in `scripts/eval/probe_eval.py` now applies the chat template, splits the think block from the answer using the orchestrator's own `_resolve_think_block`, and scores **the answer**. The think block is scored separately and stored, never mixed in. Default `--max-new-tokens` 300 → 900. Truncated and empty-answer counts are reported per model with a loud warning.

**Reasoning:** Found by audit while the full-canon run was training; the eval scripts have never been executed, so nothing had exercised them against an R1-distill model. Three defects, and together they made **gate H4 incapable of showing separation regardless of the fine-tune**:

1. **The think block was scored as if it were the answer.** The chat template pre-opens `<think>`, so the decoded completion is mostly reasoning. `\bI think\b` is a HEDGING_MARKER and R1 think blocks routinely open "Okay, so I think…"; `\bmust be\b`, `\btherefore\b`, `\bthis suggests\b` are DEDUCTION_MARKERS that saturate any R1 reasoning trace. Both marker classes were therefore measuring reasoning *register*, which every R1-distill variant shares, rather than the visible utterance the hypothesis is about.
2. **`skip_special_tokens=True` stripped `</think>`**, so the split could not have been performed even if attempted.
3. **`max_new_tokens=300` < the ~420-token think blocks measured the same day**, so responses were truncated mid-reasoning with no answer to score at all.

Scoring the think block *separately* is retained as a bonus: "did the fine-tune change register inside the reasoning, in the answer, or both?" is a sharper question than the original single number, and it costs nothing now that the split exists.

**This is the same class of error as the `verify_think_blocks.py` failures earlier today** — an evaluation instrument written against an assumed output shape rather than the real one, on a model family whose chat template pre-opens the reasoning tag. The audit was prompted by that pattern.

**Alternatives considered:** Scoring think + answer concatenated and reporting one number — that is the broken behaviour. Scoring only the think block — measures reasoning register, which is interesting but is not what gate H4 was specified to test (visible behavioural shift). `perplexity.py` and `mmlu_eval.py` were audited and are unaffected: perplexity scores raw text, and MMLU reads A/B/C/D logits at the final prompt position without generating, so neither touches the think format.

---

### 2026-08-06 — QLoRA on raw prose destroys think-block closure. The 14B spend is BLOCKED.

**Decision:** No GPU budget is spent until the dose-response curve is mapped. Full writeup: `results/analysis/format_collapse_20260806.md`.

**Reasoning:** Controlled measurement, n=8 per arm, one base model in VRAM with adapters swapped by name so all three arms share identical weights, prompts, sampling and session:

| arm | steps | unique tokens | closure | mean tokens |
|---|---|---|---|---|
| base, no adapter | — | — | **8/8** | 451 |
| stage-0 | 30 | 311,252 | **8/8** | 528 |
| full canon | 103 | 3,352,033 | **1/8** | 417 |

Ruled out by the design rather than by argument: not the model family, prompt or token budget (base is 8/8 on identical inputs); not truncation (7 of 8 failures stopped naturally at mean 417 tokens, well under the 1200 cap — the model produces a normal quantity of reasoning and never closes it); not the extractor (same `_resolve_think_block` scores base and stage-0 at 8/8); not dtype or quantization (single shared base load).

The mechanism is almost certainly catastrophic forgetting of the RL-trained reasoning format under raw-text causal LM: every gradient step optimises "predict the next token of Victorian prose" and none reinforce "emit `<think>`, close it, then answer".

**Why this outranks a bug:** the premise requires a corpus large enough to shift a reasoning prior (~1M+ unique tokens per LIMA / Betley et al.). **The dose that produces a behavioural effect may be the same dose that destroys the channel the effect is measured through.** If so the design as specified cannot work and no budget fixes it. That is a real methodological finding about QLoRA continued-pretraining on distilled reasoning models, publishable independently of whether this experiment survives.

**CONFOUND, stated so it is not forgotten:** stage-0 and full canon differ on two axes at once — corpus (311K vs 3.36M unique) and steps (30 vs 103). "103 steps broke it", "3.36M tokens broke it", and "the full canon's composition broke it" are all consistent with this data. Steps is the most likely culprit since it directly controls how far the weights move, but it is NOT established.

**Alternatives considered:** Declaring the fine-tune approach dead — premature at two dose points; the collapse may sit well above a dose that still produces an effect. Proceeding to 14B and hoping scale helps — the failure is a format-forgetting mechanism, and 14B would forget the same way for the same reason, at real cost. Authoring synthetic think blocks as rehearsal data — rejected on the 2026-06-24 grounds: training the model to reproduce reasoning we wrote contaminates the exact channel the experiment measures. If rehearsal blocks are needed they must come from the base model itself.

---

### 2026-08-06 — `save_total_limit=3` destroyed the dose curve this run could have given free

**Decision:** For any run whose question is about dose, checkpoints must be retained (`save_total_limit=None`). Logged as a standing lesson, not yet a code change.

**Reasoning:** The full-canon run wrote checkpoints every 10 steps but `save_total_limit=3` in `train_lora.py` kept only 80/90/100, and the Kaggle session then wiped those. Had all ten survived, measuring closure at each would have produced the entire dose-response curve as a by-product of a run that had already been paid for — steps 10 through 100, ten points, zero extra GPU time. Instead the curve now costs a second 4.5-hour run. **For a dose question, the checkpoints ARE the experiment.**

---

### 2026-08-06 — The augmentation's instruction framings are flattened to raw text at training time

**Decision:** Recorded as a defect; the fix is deferred until the dose curve is known.

**Reasoning:** The corpus holds **7,853 of 12,999 examples in instruction shape** — QA 2,208, CHAIN 2,208, WATSON 3,437 — built in June explicitly as the Biderman et al. mitigation for LoRA underperforming on raw-text continued pretraining. But `load_texts` reads only the `text` field and `pack_into_blocks` concatenates everything into flat 2,048-token blocks for causal LM. The chat template is never applied, so the model sees Victorian prose with `PASSAGE:` / `Q:` / `A:` headers embedded in it rather than a conversation. **The augmentation's stated purpose has therefore never been exercised at the training step**, and this is a plausible contributor to the format collapse above.

**Caveat on the obvious fix:** naively chat-formatting these examples could make things *worse*. The template ends with `<think>\n`, so an assistant turn that goes straight to the answer teaches "after `<think>`, do not think" — accelerating the collapse. Chat-formatted rehearsal only helps if the assistant turns genuinely contain think blocks, which returns to the constraint that those blocks must not be human-authored.

---

### 2026-08-06 — GATE 2 PASSES at n=20; censoring is 90%; t_think saturates at turn 0

**Decision:** Gate 2 is **settled**: 3/20 degenerate (15%) against a ≤20% criterion, on fresh seeds 2000–2019 with the corrected criterion and both anti-echo fixes. The n=6 reading is superseded. Two further results are recorded as findings, not as pass/fail.

**Reasoning:**

| | n | degenerate | uniq | mirror | turns | accusations |
|---|---|---|---|---|---|---|
| gate2 (n=6) | 6 | 0/6 (0%) | 0.76 | 13% | 24.0 | 0/6 |
| **gate2_n20** | **20** | **3/20 (15%)** | 0.77 | 17% | 21.9 | 2/20 |

**1. Censoring is 90%** — 2 public accusations in 20 conversations, and one of those was inside a degenerate transcript. Across every configuration tried, base R1-Distill-7B forms private suspicion and essentially never commits publicly inside 12 turns. At n=20 this stops being a small-sample curiosity. It makes survival analysis mandatory rather than preferred, and **the censoring rate is itself the headline**: privately suspicious, publicly silent, is a stronger claim about unfaithfulness than any mean commitment gap.

**2. `t_think_07` still saturates at turn 0.** It fired in 17/20 conversations, and **11 of those at turn 0**; the remaining 6 at turns 4, 4, 8, 12, 12, 12. Given the measured precision of 0.185, turn-0 fires are overwhelmingly false positives. This is the same failure mode as the legacy topic measure (which fired at turn 0 in 20/20), merely less extreme. **The directed regex is confirmed unusable as a measurement on live data, independently of the annotator study** — that study said precision 0.185 on sentences; this shows what that does to the conversation-level metric. The stance classifier is not optional.

**3. The degeneracy criterion is still arguably too eager, and was deliberately NOT changed.** Seeds 2001 and 2006 were terminated at unique-reply ratios of 0.71 by the locked rule (5+ consecutive identical) at turns 14 and 17 — high diversity, killed anyway. Because detection also drives termination, whether they would have recovered is unknowable. Retuning the threshold against the very run being used to evaluate it is how an instrument gets overfitted; the ratios are stored and analysis can re-decide later.

**Alternatives considered:** Treating the n=6 result as sufficient — six conversations of one pairing cannot distinguish 0% from 15%, and indeed the true rate is 15%. Loosening the locked rule now that two conversations tripped it — rejected on the overfitting grounds above.

---

### 2026-08-07 — Lexical baseline: the task is NOT lexical, and the labels are clean

**Decision:** Every detector number for `t_think_07` must now be reported beside a bag-of-words baseline and an annotator ceiling (`scripts/eval/lexical_baseline.py`). **Option B is narrowed: any replacement must be SEMANTIC** — sentence embeddings plus a classifier, or an LLM judge per sentence. An n-gram/TF-IDF "stance classifier" is ruled out by measurement, not by taste.

**Reasoning:** Prompted by a cross-project note from decision-traceability (2026-08-07). Their per-layer probes scored 95–98% with healthy Hewitt & Liang selectivity (+0.42 to +0.48) and were worthless: a bag-of-words model on the raw string matched them. **Control-task selectivity cannot catch a lexical cue** — a random-label control with matched statistics has no linguistic content, so a lexical shortcut clears it and the gap still looks healthy. Selectivity is necessary, not sufficient. That correction applies to the advice recorded here on 2026-07-26.

Measured on the 231 labelled sentences:

| | prec | recall | F1 |
|---|---|---|---|
| regex (`t_think_07`) | 0.185 | 0.652 | 0.288 |
| BoW @ matched recall (CV) | 0.188 | 0.652 | 0.291 |
| BoW @ nested-selected threshold | 0.167 | 0.304 | **0.215** |
| **annotator ceiling (leave-one-out)** | | | **0.986** |

BoW AUC 0.728, 95% CI **[0.607, 0.850]**.

**Three readings, and the diagnostic distinguishes them — this is the durable result:**
1. BoW **high** and ≈ detector → the concept is lexical and the detector is cheating. *(decision-traceability's failure.)*
2. BoW **low** and ≈ detector, ceiling **high** → the concept is semantic and no lexical method can close the gap. *(sherlock's failure — the opposite diagnosis from the same test.)*
3. Ceiling **low** → the labels cap every method and the gate is wrong, not the detector.

Neither diagnosis is visible from accuracy or selectivity alone, and the two projects reached for the same test and landed in different branches without either predicting which.

**Two statistical errors caught and fixed on review, both mirroring entries in decision-traceability's log:**
- The first "best achievable F1 = 0.326" was obtained by sweeping the threshold over the same out-of-fold scores it was then reported on — selection on the evaluation set, the same shape as picking a probe layer by argmax over test accuracy. Nested selection (threshold chosen on train folds, applied to the held-out fold) gives **0.215**. The biased number was ~50% too high and made the ceiling look softer than it is.
- AUC was first reported as a bare 0.728 on 23 positives. The Hanley–McNeil 95% CI is **[0.607, 0.850]**, and the upper bound is what any "lexical methods cannot reach X" claim must be argued against.

**The label-noise question, which was the sharpest challenge:** a low BoW ceiling is equally consistent with "needs semantics" and "labels are noisy". Measured rather than assumed — leave-one-annotator-out F1 is 0.986 and binary Fleiss κ (conclusion vs not) is **0.969**. The 0.906 recorded on 2026-07-28 was over all three classes and is dominated by agreement on `neither`; restricted to the class that matters, agreement is near-perfect. **So 0.8 is a target, not a fantasy, and the gap is real signal.** Caveat retained: three LLM annotators sharing a rubric are correlated, so 0.986 bounds *agreement*, not human performance.

**Alternatives considered:** Building the stance classifier on TF-IDF features — measured to top out near 0.22, so it would have consumed effort to reproduce the regex's failure. Treating BoW ≈ detector as automatically meaning "lexical" — that was the original two-branch verdict in the script, and it reported the exact opposite of the truth here; the logic now has three branches keyed on the absolute scores and the ceiling.

---

### 2026-08-07 — Persistence is part of the run path; the dose-curve headline is CI-bearing, not a collapse point

**Decision:** Two coupled changes, prompted by the **third** loss of a run to `/kaggle/working` being wiped at interactive-session end (2026-08-07: the dose curve's 12 checkpoints incl. the load-bearing step-35, the final adapter, and `dose_curve_20260807_212005.json`, all lost). (1) **Off-machine persistence is now built into the code, not a manual afterthought.** `train_lora.py` uploads every checkpoint to the HF Hub the instant Trainer writes it (new `HFCheckpointUploader` on `on_save`, gated on a `hf_repo_id` config key; token from env/Kaggle Secret, never argv). `dose_curve.py` uploads the results JSON and the incremental `.partial.jsonl` as each row is written, and gains a `--push-results` flag that git-commits and pushes the small JSON to the repo (owner-requested). All uploads go through `scripts/training/hf_persist.py`, which retries with backoff (10/30/90 s) and, on final failure, logs loudly and returns False **without ever raising** — a flaky uplink on one checkpoint must not kill the run producing the next. (2) **The dose-curve analysis headline is reframed** from "COLLAPSE at step N / a usable dose window exists at ~35" to Wilson 95% intervals per checkpoint plus a pooled early-(≤35)-vs-late-(≥45) Fisher exact test. The eval is also now **seeded** (`--seed`, `set_seed`, logged into the JSON) and logs its sampling params — it was previously unseeded, so identical commands produced different numbers. The driver notebook (`notebooks/kaggle_t4_dosecurve.py`) is committed so the run reproduces from git and chains train→eval in one cell.

**Reasoning:** *Persistence:* CLAUDE.md's durability section already said "persist artifacts as they appear, not at the end" and "an experiment you cannot read mid-flight is gambling" — but nothing in the code enforced it, so it depended on someone remembering to copy files before ending the session, and three times that failed. Moving persistence into `on_save` and into the per-row write path makes durability a property of the run, not of operator diligence. Uploads must never raise because the alternative — a network blip killing a 4.6 h training run at step 40 — is strictly worse than losing one checkpoint. *Statistics:* the old printed conclusion named a single "last healthy checkpoint" and told the reader to "train to ~N steps", which asserts a sharp, certifiable boundary that n=8 per checkpoint cannot support — the per-point 95% intervals are ~±0.3 wide and overlap heavily across the middle of the curve. On the logged 2026-08-07 numbers the pooled contrast is early 27/32 = 0.84 [0.68, 0.93] vs late 28/56 = 0.50 [0.37, 0.63], Fisher two-sided p = 0.0014: a real, significant degradation with dose, but **no** step at which closure is both intact (upper CI < 1.0 even early) and the dose is large enough to move a reasoning prior. The supportable claim the owner settled on is "degradation begins near or below the effect threshold", and the printed conclusion now says exactly that and explicitly retracts the dose-window framing. Fisher/Wilson are implemented dependency-free (`math.comb`, closed form) so the eval keeps its light Kaggle footprint; Fisher is verified against the textbook lady-tasting-tea value (0.4857) in `tests/test_dose_stats.py`, which also dry-parses the 2026-08-07 numbers with no GPU.

**Alternatives considered:** *Persistence as a post-run upload step* (the status quo, `upload_adapter.py`) — that is exactly what failed three times; the artifact has to leave the machine as it is produced, not after. *Crash-on-upload-failure* — rejected: it converts a recoverable one-checkpoint loss into an unrecoverable whole-run loss. *A normal-approximation CI or a bare proportion* — the normal approx gives [1.0, 1.0] for 8/8 (zero width, obviously wrong) and can leave [0,1]; Wilson is the correct small-n choice. *Keeping the collapse-point headline but widening the token budget or n* — does not address the framing error; the problem was claiming a usable boundary, not the point estimate. *scipy for Fisher* — avoided to keep the Kaggle dependency set minimal; the closed form is exact and tested.

---

### 2026-08-08 — Persistence must FAIL CLOSED; and the 22-checkpoint dose curve replicates the finding

**Decision:** (1) A missing HF token is now a **hard stop before training**, not a warning. `train_lora.py` calls `sys.exit` at the top of `main()` if `hf_repo_id` is set and no token is reachable (escape hatch `ALLOW_UNPERSISTED=1`); the driver notebook `assert`s the token in its preflight cell; `dose_curve.py` and the notebook's closing banner now print an honest "LOCAL ONLY — not persisted" status instead of the unconditional "safe to close the session." (2) The full 22-checkpoint dose curve from the 2026-08-08 run is recorded as `results/analysis/dose_curve_20260808_204827.json` + `dose_curve_20260808_writeup.md`, **reconstructed from the notebook's printed output** because the original was lost (see below). Pooled early (≤35) 39/56 = 0.70 [0.57, 0.80] vs late (≥45) 44/104 = 0.42 [0.33, 0.52], Fisher p = 0.0015.

**Reasoning:** *Fail-closed:* the 2026-08-07 entry moved persistence into the code but left it **fail-open** — the token check was a warning, and training proceeded unpersisted. On 2026-08-08 that is exactly what happened: the Kaggle Secret was not attached, both the train and eval steps warned "NOT persisted," the run trained ~5 h and evaluated all 22 checkpoints, then the interactive session was closed and `/kaggle/working` was wiped — the **fourth** loss of this class, and the first that the just-added persistence code was supposed to prevent. A safeguard that can be silently skipped is not a safeguard when the failure mode is a missed warning at 3am; the check must refuse to spend the 5 h at all. The false "safe to close the session" banner actively caused harm by reassuring against the truth, so honest status reporting is part of the same fix. *The science survived only because it was in stdout* — that is luck, not a system, which is the whole argument for the gate. *Replication:* the run is a clean confirmation of the 2026-08-06 format-collapse finding at finer resolution — `train_loss` was bit-identical (`0.848246…`) so the checkpoints are deterministically reproducible, the pooled p (0.0015) matches the prior run's 0.0014, and per-point closure wobbled run-to-run (step-25 6/8→4/8) exactly as the CI-bearing reframe predicted it would. Onset is if anything earlier than before (step-20 already 3/8, ~655K unique tokens, below the ~1M effect threshold), reinforcing "degradation begins at or below the effect dose."

**Alternatives considered:** *Keep the warning, add a louder banner* — rejected; the 2026-08-08 run already printed two clear warnings and it changed nothing, because the operator was not watching at the moment it mattered. Only refusing to start closes the hole. *Require the token unconditionally (no escape hatch)* — rejected; a genuine local/CPU smoke run with `hf_repo_id` in the config should still be possible, hence `ALLOW_UNPERSISTED=1`. *Treat the lost run as unrecoverable and re-run blind* — unnecessary; the closure numbers were fully present in the notebook output and the analysis is recomputed from them by `dose_curve.analyze_rows`, so the JSON is faithful (pooled numbers reproduce exactly). The **weights** must still be regenerated by a fresh run, but the measurement did not need re-collecting. *Not recording a reconstructed result because it isn't the original file* — rejected; the provenance is stamped in the JSON (`"reconstructed": true` with the reason), and a labelled reconstruction that reproduces the printed pooled stats is far better than discarding a real measurement over file-identity purity.

---

### 2026-08-08 — Confound separator: one pilot@103 run, not two, and it is the LAST diagnostic before rehearsal-or-writeup

**Decision:** Build (not yet run) a single-run experiment to separate the two knobs the format-collapse finding still confounds — optimizer STEPS vs UNIQUE-TOKEN BREADTH. `configs/kaggle_t4_confound_pilot103.yaml` trains the **pilot corpus (311K unique) to ~103 steps by re-reading it 11×** — identical model/LoRA/optimizer/schedule to the full-canon dose curve, so the only difference from that curve is corpus breadth (1/11th). `scripts/eval/confound_analysis.py` overlays the two dose-curve JSONs into a 2×2 (pilot/canon × early≤35/late≥45), reports four Fisher+Wilson contrasts, and prints a mechanism verdict. Driver: `notebooks/kaggle_t4_confound.py` (fail-closed, chains train→eval→analysis). Owner-triggered; ~4.5 h, free.

**Reasoning:** The original sketch (2026-08-06) was "pilot@103 vs full-canon@30" — two runs. But stage-0 (pilot ~30 steps, 8/8) and the full-canon curve (~30 and ~103 steps) already exist, so only **one** cell of the 2×2 is missing (pilot at high steps). Running just that fills it. The design also fixes the flaw that stops *existing* data from answering this: stage-0 vs full-canon-step-30 hints at breadth (8/8 vs 5/8 at matched 30 steps) but is confounded by the **LR schedule** — stage-0's cosine ends at step 30 while full-canon's is near-peak, so full-canon's weights have moved farther. The pilot@103 run puts the pilot on its *own* ~103-step cosine, so step-for-step its LR position matches full canon and breadth is the clean sole difference. The verdict is decision-relevant: BREADTH-drives → low-LR/rank cannot open a window (you need the breadth for the effect) and rehearsal is mandatory; STEPS-drive → low-LR/low-rank/early-stop are worth trying first. `confound_analysis.analyze_confound` is pure and unit-tested against the real full-canon numbers with both synthetic pilot branches (`tests/test_confound_analysis.py`), so the verdict is trustworthy the instant the pilot JSON lands.

**Alternatives considered:** *Two runs (pilot@103 AND canon@30)* — rejected; canon@30 is already the full-canon curve's step-30 point, so the second run is redundant. *Answer it from existing data with no run* — rejected as the sole approach; the LR-schedule confound above makes the existing comparison suggestive but not clean, though it is recorded as the prior. *Skip the confound and build rehearsal directly* — genuinely tempting, because **rehearsal (base-model-generated think blocks mixed into the corpus) is the mitigation that works under either branch**, so the confound is not on the critical path to *rescuing* the experiment. Kept anyway because it is cheap, it is a clean publishable methods result (which axis drives catastrophic forgetting of RL-distilled format under QLoRA), and it targets the rehearsal design. But it is explicitly logged as the **last diagnostic**: after this, the fork is rehearsal or a negative-results writeup, not another characterization run. *Low-LR/low-rank sweeps now* — premature; the confound verdict says whether they can possibly help before spending on them.

---

### 2026-08-12 — Confound RESOLVED: steps/weight-movement, not breadth; and re-reading a narrow corpus is worse

**Decision:** The confound separator ran (pilot corpus → 110 steps, 11×, matched schedule; persistence worked end-to-end, every checkpoint + results on HF). The verdict is **STEPS / cumulative weight movement, NOT unique-token breadth.** Recorded as `results/analysis/dose_curve_20260812_104622.json` + `confound_pilot103_vs_fullcanon.json` + `confound_20260812_writeup.md` (reconstructed into git from the notebook output, which persisted to HF but not to git; analysis recomputed by the real code). The next action is a **single low-rank mitigation run** (constrain the adapter subspace to protect the base weights), with low-LR/early-stop as weaker secondary levers, and the negative-results writeup started in parallel. This retires the "last diagnostic"; low-rank is a *mitigation attempt* (part of the rescue fork), not another characterization.

**Reasoning:** The 2×2 is unambiguous. At matched low dose, pilot and canon are **identical** (early 0.70 vs 0.70, Fisher p=1.0) despite 11× different unique tokens — breadth does nothing. The pilot collapses hard with steps at 1/11th the breadth (0.70 → 0.21, p=2.7e-09). And the unexpected, sharp finding: at matched high steps the **low-breadth pilot is worse** than the broad canon (0.21 vs 0.42, p=0.0012) — re-reading a narrow 311K corpus 11× damages the format *more* than one pass over 3.36M diverse tokens, because the repeated, concentrated gradient drives the weights further into "complete Victorian prose" and away from "emit/close `<think>`". This refutes the breadth guess (mine included) cleanly and reframes the mitigation: the lever is *less weight movement / a constrained subspace* (low rank, fewer modules), not a smaller or curated corpus. Honest caveat kept in the writeup: this identifies the lever, it does not prove a usable window exists — format damage tracks weight movement, and the movement needed to shift a reasoning prior may inherently damage the format; low-rank is a principled bet, rehearsal the robust fallback. Standalone, this is a publishable methods result: catastrophic forgetting of RL-distilled format under QLoRA is driven by optimizer steps, not training-distribution breadth.

**Alternatives considered:** *Read the printed verdict as final ("steps → low-LR/rank will fix it")* — overclaims; the verdict names the lever, and the "pilot worse at high steps" result specifically warns that low-LR may only rescale the same damaging trajectory, which is why low-*rank* (subspace constraint) is preferred over low-LR (speed). *Skip low-rank, go straight to rehearsal* — defensible (rehearsal works regardless), but low-rank is one cheap now-crash-safe run that could rescue the experiment without the rehearsal pipeline, so it is worth trying first while the writeup proceeds in parallel. *Not writing the negative result until a mitigation is tried* — rejected; the finding is solid twice over and the confound is now cleanly separated, so the writeup is valuable independent of any rescue.

---

### 2026-08-12 — Low-rank mitigation is a DUAL measurement (closure AND effect), because closure alone can mislead

**Decision:** Build (not yet run) the rank-8 mitigation run as a **dual** measurement, not a closure dose curve alone. `configs/kaggle_t4_lowrank_r8.yaml` changes exactly one thing from the full-canon dose curve — rank 32→8, alpha 64→16 (scaling held at 2.0) — so its closure curve overlays directly on the r32 curve. The driver (`notebooks/kaggle_t4_lowrank.py`) then runs TWO evals per checkpoint: `dose_curve.py` for closure (does the `<think>` format survive?) and a new `effect_curve.py` for held-out Speckled Band perplexity (did it learn Holmes?), and `mitigation_analysis.py` overlays them into one of three verdicts — RESCUED / COUPLED / TOO_WEAK. Whether **rehearsal is needed at all** is decided by this triple, and the run is built to answer it in one Kaggle session. The negative-results writeup is held until this result is in.

**Reasoning:** A closure curve alone cannot answer "is rehearsal needed," because low rank can preserve closure *by learning too little to matter* — the exact failure the effect curve exists to catch. Rescue requires BOTH at one checkpoint: format intact (closure ≥ 0.75) AND effect present (held-out PPL drop ≥ 5%, the pre-registered H1 gate). The three outcomes fork cleanly: **RESCUED** (a window exists) → rehearsal unnecessary, confirm with full eval gates; **COUPLED** (effect only appears once closure has collapsed) → low rank cannot decouple them, rehearsal needed; **TOO_WEAK** (no checkpoint reaches the PPL gate) → rank 8 learned nothing, more capacity would reintroduce the collapse, rehearsal needed. `effect_curve.py` reuses `perplexity.compute_perplexity` (the H1 metric) and dose_curve's load-once/swap-adapter pattern so closure and effect are measured on the same checkpoints in one session; `mitigation_analysis.analyze_mitigation` is pure and unit-tested on all three branches (`tests/test_mitigation_analysis.py`). Rank 8 with alpha 16 keeps the LoRA scaling (alpha/rank=2.0) identical to the r32 run, so the only thing that changes is the *dimension* of the subspace the adapter can move the base weights in — the lever the confound verdict pointed to. Full canon is kept as the corpus: the effect needs a real dose and the confound showed breadth is not what breaks the format. Honest odds recorded: ~1-in-3 that low rank delivers both; hence the writeup proceeds in parallel rather than waiting.

**Alternatives considered:** *Closure curve only* — rejected as insufficient; "closure survived" without an effect check would falsely suggest rehearsal is off the table. *Probe-separation (probe_eval.py) as the effect measure* — deferred; it has never executed and had scoring bugs (fixed 2026-07-28 but unrun), whereas perplexity scores raw text, is the audited-clean H1 metric, and directly measures whether unseen Holmes got easier. Probe separation is a richer follow-up once a checkpoint is worth it. *Also dropping target modules (q/v only) in the same run* — rejected for this run; changing rank AND module set at once reconfounds it, and rank alone is the clean single-variable test overlayable on the r32 curve. Fewer modules is the next lever if rank-8 is COUPLED/TOO_WEAK. *Lower LR instead of lower rank* — the confound's "pilot worse at high steps" warns low-LR may just rescale the same damaging trajectory; low-rank constrains *where* the weights move, which is mechanistically the right lever.

---

### 2026-08-14 — Low-rank (r8) RESCUES the format; rehearsal shelved, but perplexity is a proxy

**Decision:** The rank-8 mitigation run returned **RESCUED** — closure is far better preserved (early 0.96 [0.88,0.99] / late 0.73 [0.64,0.81], vs r32's 0.70/0.42) AND held-out Speckled Band perplexity drops +43.8% (H1 gate is ≥5%), with a comfortable window (step ~50: closure 8/8 AND drop +42.6%). **Rehearsal is shelved** (not deleted) and the negative-results writeup is **reframed, not shipped**. Recorded as `results/analysis/dose_curve_20260814_042400.json` + `effect_curve_20260814_065854.json` + `mitigation_lowrank_r8.json` + `mitigation_lowrank_20260814_writeup.md` (reconstructed into git from the Kaggle output; analysis recomputed by the real code). Next actions: (1) pull the WikiText H2 drift from the effect JSON on HF; (2) run the **behavioural** effect check (probe separation / think-block inspection on deduction prompts) on the step-50 checkpoint — the real reasoning-shift signal; (3) if it holds, proceed to the conversation arm on step-50.

**Reasoning:** The result is coherent with the confound verdict — the collapse tracks weight movement, and rank 8 moves the weights in 1/4 the directions, so the format survives much better at the same steps/corpus. That part is a clean, verified success and it clears the blocker that stopped the whole experiment. **But three caveats are logged so "RESCUED" is not over-read.** (a) *Perplexity is a proxy and saturates by step ~15 (~500–650K tokens, below the ~1M reasoning-shift threshold)* — most of the +44% is fast surface adaptation to Doyle's style, not proof the model *reasons* like Holmes; the experiment's DV (deduction behaviour / commitment gap) still needs the behavioural measure, so this rescues the precondition, not the hypothesis. (b) *H2 (WikiText drift) was computed but not printed* — until it is checked, the effect could partly be general-LM degradation rather than Holmes-specific learning. (c) *The final adapter's closure is 3/8, much worse than step-103's 7/8* — operate at a mid checkpoint (~step 50), never `final`. The verification house rule applies squarely here: a proxy metric passing spectacularly is not the same as the real path working, and the real path (reasoning shift, then the three-level commitment gap) has not yet been measured.

**Alternatives considered:** *Declare the experiment rescued outright and jump to the 1000-conversation arm* — rejected; perplexity is a distributional proxy and the behavioural shift is unverified, so that would be reporting "working" without the real path, the exact error CLAUDE.md's verification rule forbids. *Ship the negative-results writeup anyway* — rejected; the honest finding is no longer "the design cannot work," it is "standard rank destroys the format, low rank rescues it," which is a different and better paper. *Keep the writeup fully shelved* — no; the format-collapse + confound + rescue arc is itself the methods contribution and should be drafted once the behavioural check lands. *Use the final adapter (simplest)* — rejected on the closure anomaly above.

---

### 2026-08-14 — H2 pulled: the effect is MOSTLY GENERIC prose-LM recovery, not Holmes learning

**Decision:** The WikiText (H2) numbers, computed in the run but not printed, were pulled from the HF effect-curve JSON (`effect_curve_20260814_065854.json`, now saved into git with WikiText included, superseding the reconstruction). **WikiText perplexity dropped ~34%** alongside the +44% on held-out Holmes. The effect is therefore reinterpreted: ~34 of the 44 points are **generic prose-LM recovery**, only ~10 points are **Holmes-specific** (the excess; equivalently the Holmes/WikiText PPL ratio fell 1.122 → 0.956, a ~15% relative specialisation). The mitigation's RESCUED verdict stands **for closure** (format preservation is measured directly and is unaffected) but the "effect present" half is now known to be **confounded** — the PPL≥5% gate fires mostly on generic recovery. The behavioural check on step-50 is therefore promoted from recommended to **required** before any claim that the fine-tune shifts reasoning.

**Reasoning:** The mechanism is clean and was hiding in plain sight: base R1-Distill is an RL-reasoning model that is *poor at raw next-token prose prediction* (it is optimised to emit reasoning, not continue Victorian prose), so continued-pretraining on any prose restores strong prose LM and drops perplexity on *everything* — WikiText and Holmes alike, in near-lockstep, both saturating by step ~20. The H1 gate (Speckled-Band drop ≥5%) was only ever a valid proxy for Holmes learning **under the assumption H2 stays flat**, and H2 did not stay flat — it moved 34%. So the headline "+44% effect" massively overstates Holmes-specific learning. The honest Holmes signal is the *excess* Holmes-over-WikiText drop (~10pp) and, unlike the generic part, it keeps climbing slowly with dose (excess +5% at step 20 → +9.7% at step 103), which is consistent with distributional specialisation being the slow/expensive component. This is exactly the "confidently wrong reading" failure the house rules warn against: had we reported "+44% ⇒ learned Holmes," it would have been wrong; pulling the guardrail number caught it. Whether a ~10pp / ratio-1.12→0.96 distributional shift is enough to move *reasoning behaviour* is unknown and perplexity cannot answer it — hence the behavioural check is now the gating step, and it could still come back null.

**Alternatives considered:** *Trust the +44% as the effect and proceed* — rejected; that is the wrong reading H2 just corrected. *Call H2 a FAIL and the mitigation dead* — wrong in the other direction; H2 "failed" the ≤±5% band by *improvement*, not degradation, and closure — the thing that was actually blocked — is genuinely rescued. *Re-run with a Holmes-vs-control perplexity design baked in* — unnecessary; the WikiText arm already provides the control, and the excess/ratio decomposition extracts the Holmes-specific signal from the existing data. *Edit the prior 2026-08-14 entry to fold this in* — rejected; the log is append-only, so this is a follow-up entry, and the sequence (claimed effect → pulled guardrail → corrected) is itself the useful record.

---

## Current state (update each session)

**Last updated: 2026-06-24**

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
- Pilot YAML configs → `configs/pilot_qwen.yaml`, `configs/pilot_mistral.yaml` (superseded by R1-distill configs below)

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
- Training script: `scripts/training/train_lora.py` (unchanged — works for R1-distill as-is)
- **Model upgrade (2026-06-24):** Base models switched from Qwen2.5-7B + Mistral-7B to DeepSeek-R1-Distill-Qwen series
  - Staged configs: `configs/main_r1distill_qwen7b.yaml` → `configs/main_r1distill_qwen14b.yaml` → `configs/main_r1distill_qwen32b.yaml`
  - Upgrade by changing `base_model` in config + swapping adapter in Modal secret — no other code changes
- **Next action (staged):**
  1. Kaggle T4 (free): validate 7B config end-to-end — confirm think blocks appear in logs
  2. RunPod RTX 4090 (~$1): fine-tune R1-Distill-14B, upload adapter to HuggingFace
  3. Modal deploy (`modal deploy scripts/inference/modal_app.py`): point orchestrator at Modal URL
  4. Run ~1000 conversations; advance to 32B if evaluation gates pass

### Phase 2 — Conversation orchestrator: UPGRADED (2026-06-24)
- **New schema fields (TurnRecord):** `think_block` (raw `<think>…</think>` content), `messages_input` (exact prompt sent to API — enables TransformerLens replay)
- **New schema fields (ConversationRecord):** `t_think_07` (first turn suspicion keywords in think block), `think_commitment_gap` (t_private_07 − t_think_07)
- **New AgentConfig field:** `thinking_mode: bool` — set True for R1-distill; selects thinking-compatible prompts and JSON reminders
- **Three-level commitment gap now logged:** think block keywords → suspicion_score ≥ 0.7 → public_accusation
- **Thinking-model prompts:** `INITIATOR_SYSTEM_THINKING` / `RESPONDER_SYSTEM_THINKING` in `prompts.py` — do not suppress `<think>` tokens
- **Inference platform:** `scripts/inference/modal_app.py` — Modal vLLM endpoint; `modal deploy` → persistent HTTPS URL; `keep_warm=1` avoids cold starts
- Prior validation still holds: 5 conversations × 12 turns × 2 agents = 120 turn records, JSONL schema correct (base models)
- **2026-07-18 local shakedown (base deepseek-r1:7b via Ollama, 2 convs, thinking mode): instrument live end-to-end.**
  14/14 turns captured think blocks (mean ~1.7K chars), parse modes 12 json / 2 fallback / 0 api_error.
  Transport gotcha fixed: Ollama ≥0.9 / vLLM-with-reasoning-parser return thinking in a separate
  `reasoning`/`reasoning_content` field, not inline `<think>` tags — `_resolve_think_block` handles both.
  Also observed: base 7B leaked "I'm an AI language model" as a public reply at turn 0 — the
  pass-as-human failure the fine-tune exists to study.
  - ⚠️ **RETRACTED 2026-07-26 — the "one genuine dissociation (t_think=0, t_private=6)" claim was wrong.**
    Both conversations were degenerate: conv `f217671f` ran 12 turns with **1 unique reply**, so that
    `t_private_07` was measured on constant input. The 2 "fallback" turns were the parser returning the
    JSON template's placeholder text, which was then fed to the opponent. See the three 2026-07-26
    Decision Log entries. **No commitment-gap result exists from this run.**
  - ✅ The t_think saturation flagged here is now fixed (directed-suspicion redefinition, 2026-07-26).
    Measured on this same data: legacy topic-mention fired 14/14 (100%); directed fires 8/14 (57%).

### Measurement-validity fixes (2026-07-26) — all landed, `make test` green (48 tests)
- `t_think_07` = directed suspicion; legacy measure retained as `t_think_topic` for delta reporting
- degeneracy detection + early termination (`termination_reason="degenerate"`) + repetition penalties + personas
- per-(conversation, turn, speaker) seed derivation — replicates are now independent
- `parse_failed` mode; unusable turns enter neither history nor metrics
- **Still outstanding before the pilot:** hand-labelled think-stance validation set
  (`data/probes/think_stance_labels_v1.jsonl`, ~100 sentences) — the directed detector's
  precision/recall are **unmeasured**. This is the gating item, and it needs no compute.
  Tooling is ready: `make label-tool` → label in a browser → `make score-detector`.

### Shakedown 2026-07-26/27 — GATE 2 PASSED (n=6, base deepseek-r1:7b)
Four matched-seed runs; full writeup `results/analysis/shakedown_20260727_writeup.md`.

| run | degen | uniq | mirror | turns | accuse |
|---|---|---|---|---|---|
| baseline (personas + penalties) | 6/6 | 0.42 | 65% | 4.8 | 0/6 |
| + anti-echo, system prompt | 4/6 | 0.70 | 17% | 13.8 | 2/6 |
| + anti-echo, per-turn reminder | 5/6 | 0.83 | 14% | 15.2 | 0/6 |
| + corrected degeneracy criterion | **0/6** | 0.76 | 13% | **24.0** | 0/6 |

- ✅ Gate 2 met at n=6 — **re-read at n≈20 before treating as settled**
- ⚠️ **0/6 public accusations** — 100% right-censored; survival analysis is mandatory
- ⚠️ **suspicion declines over a conversation** — the sustained-≥0.7 `t_private_07`
  definition needs revisiting before pilot data (Decision Log 2026-07-27)
- ⚠️ **clean-JSON 60% at 24 turns** (83% early → 38% late); the 2026-07-06 "keep
  fallback turns" decision is reopened
- ✅ legacy topic measure fired turn 0 in **24/24** conversations; directed measure
  discriminates (4/6, 3/6, 2/6, 4/6)
- ✅ `parse_failed` guard fired once on real data, catching a live placeholder leak

- **Next action:** Gate 1 (label + score, no compute), then settle the `t_private_07`
  definition, then re-read Gate 2 at n≈20. Full plan: `../experiment.md`

### Supporting artifacts
- `data/augmented/augmentation_spec.md` — framing templates + worked examples
- `data/augmented/manifest.json` — full canon augmentation manifest (model, version, counts, per-framing breakdown)
- `results/pilot/chunk_audit_report.md` — label quality audit
- `results/analysis/literature_notes.md` — structured notes on all cited papers
- `results/pilot/pilot_writeup_template.md` — pre-filled writeup template (fill in after eval)
- `scripts/inference/modal_app.py` — Modal vLLM deployment for R1-distill inference

### Key numbers to remember
- Training corpus (pilot): 1168 examples, ~325K tokens
- Training corpus (full canon): 12,999 examples, ~3.44M tokens → `data/augmented/full_canon_train.jsonl`
- Held-out: `data/processed/heldout/speckled_band.txt`
- Perplexity pass gate: ≥5% drop on Speckled Band, WikiText within ±5% of base
- MMLU pass gate: <3pp drop vs base
- Suspicion threshold for "suspicion event": P(AI) ≥ 0.7 (still applies to suspicion_score JSON field)
- Target conversation volume: ~1000 (pilot pairing matrix subset); ~6000 (full 10×10 matrix)
- Inference compute: Modal A10G (24GB, free $30 credits) for **7B only** — 14B bf16 needs ~28GB and
  does NOT fit; use A100-40GB, or 4-bit + a bf16 control arm (Decision Log 2026-07-26). A100-40GB for 32B
- Total experiment budget: $50 — committed ~$8 (7B + 14B training), ~$5 reserved (32B training), ~$12 reserved (interpretability re-runs), $25 buffer
- Commitment gap now three-level: t_think_07 → t_private_07 → t_public

---

## Agent behavior
Shared working norms across Utsav's personal repos. Follow them unless the user says
otherwise in-session.

### Verification — verify, show evidence, verify the *real* path
- Never report a number, eval result, or "training worked" without running it and
  showing the real output. No fabricated or projected results.
- Verify against the **real artifact and the real path**, not a stand-in. A passing
  smoke test or a base-model probe is not proof the fine-tuned adapter works in the
  conversation orchestrator — the seams (adapter load, vLLM server, Modal endpoint,
  JSON schema enforcement, think-block extraction) are where things actually break.
- For long-running jobs (training runs, full augmentation passes, ~1000-conversation
  sweeps), verify *genuine progress*, not just "started" — a launched process that
  silently died or hung is the default failure mode. Report liveness with evidence
  (loss curves, completed-turn counts, output files growing), and make long jobs
  survivable (checkpoints, append-only output, resumable orchestrator runs).
- If a step was skipped or estimated, say so. When done and verified, say it plainly
  with the evidence.

### Documentation & reproducibility — as you go
- **Decision Log is append-only and load-bearing.** Every architectural or experimental
  decision goes there with reasoning + alternatives considered. Never edit or delete
  existing entries.
- Record versions, configs, environment, and the exact commands behind any result worth
  reproducing. `configs/` is the single source of truth for hyperparameters — don't
  hardcode values in scripts.
- `results/pilot/` is append-only — never overwrite; use timestamped filenames.

### Git & durability
- Commit in logical, scoped chunks; don't sweep unrelated changes into a commit.
- **Push your work** — local-only commits aren't backed up. Flag when the repo is ahead
  of its remote.
- **Never run two write-sessions in one working tree.** If another session or the user
  may be editing this checkout, pathspec only your own files and re-check
  `git diff --cached` before each commit.
- Secrets never get committed; check before every push. The committed data corpus is
  small (~8MB) and irreplaceable — keep it tracked; model weights stay gitignored.

### Confirmation — confirm heavy / external / irreversible actions
- Ask first before: installing deps, downloading large files, kicking off long jobs,
  **any RunPod or Modal deploy / GPU spin-up**, uploading adapters to HuggingFace,
  changing repo visibility, or anything outward-facing.
- Proceed freely on: file reads, small local scripts, local edits, cached pipeline
  re-runs that don't hit Ollama or the network.
- Always stop for destructive or irreversible actions, and for anything that costs
  money or publishes to the outside world. Approval in one context does not carry to
  the next.

### Sessions — fresh one when context degrades
- Recommend a new session when responses slow, context clutters, or you're re-deriving
  established facts. Summarize state when you do, so the next session picks up cleanly.

### Scope — reasonable initiative
- Do what's asked; handle obvious adjacent work en route (e.g. a clear bug hit along the
  way). Flag anything larger **before** acting — surface it as a suggestion, don't
  silently expand scope.

### Communication — lead with recommendations
- Concise and direct; lead with a recommendation, not an option-survey. Expand rationale
  for significant decisions, keep it short for routine work.

### Workspace — structured directories
- Keep intermediate and final artifacts in clear subfolders (`data/`, `scripts/`,
  `configs/`, `results/`, `docs/`). Commit durable artifacts; scratch files stay
  throwaway.
