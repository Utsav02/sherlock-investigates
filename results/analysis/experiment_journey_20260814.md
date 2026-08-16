# Sherlock Investigates — Experiment Journey & Results

**As of 2026-08-14.** A from-scratch record of what we set out to measure, every
result we got, what each result forced us to do next, and where that leaves us.
Written as the backbone for an eventual writeup. Per-result detail lives in the
sibling files in `results/analysis/`; this is the thread that connects them.

---

## 1. The question we set out to answer

The novel measurement is the **commitment gap**: in adversarial "spot the AI"
conversations — each agent trying to identify the other as human or AI while
passing as human itself — *how long is an agent privately suspicious in its
chain-of-thought before it commits to that suspicion out loud?* With reasoning
models that emit explicit `<think>` blocks, this is a **three-level** measure per
agent:

1. `t_think` — first turn the private think block asserts a directed suspicion,
2. `t_private` — first turn the JSON `suspicion_score` crosses 0.7,
3. `t_public` — first turn the agent publicly accuses.

The gap between these is a measure of **unfaithfulness** — visible behaviour
dissociating from hidden reasoning. That is the contribution.

The experiment has two phases:

- **Phase 1 — fine-tune** small open models on different corpora to produce
  variants with *distinguishable reasoning priors* (e.g. a Holmes-style
  "confident deducer" vs base vs controls). This is the *manipulation*.
- **Phase 2 — converse**: pair the variants in adversarial conversations and
  measure the commitment gap.

**Everything in this document is Phase 1.** We never reached Phase 2, and the
reason we never reached it is the whole story below.

---

## 2. Setup & stack

- **Base model:** DeepSeek-R1-Distill-Qwen-7B (chosen for explicit `<think>`
  blocks + free-GPU trainability + published interpretability priors). Staged
  plan 7B → 14B → 32B; **we never got past 7B.**
- **Corpus:** the full Conan Doyle canon. 12,999 augmented examples,
  **3,352,033 unique tokens** (measured). Held-out: *The Speckled Band*
  (excluded from training, used to test generalisation). A smaller **pilot
  corpus** (311,252 unique tokens) was used for plumbing.
- **Method:** QLoRA (4-bit NF4), originally rank 32 / alpha 64.
- **Infra:** Kaggle T4 (free) for training; HuggingFace for persistence; Modal
  planned for inference. **Budget cap $50; actual spend ≈ $0** — everything ran
  on free Kaggle and local Ollama.

---

## 3. Timeline: what we did → what we found → what it forced next

### 3.0 Instrument & measurement work (before any GPU)

Long before training, we built and hardened the measurement so a result would be
*interpretable*:

- **Conversation orchestrator** with a single structured-JSON call per turn
  (`reply`, `suspicion_score`, `reasoning_trace`, `public_accusation`, …), and
  the three-level commitment-gap logging above.
- **Measurement-validity fixes:** directed-suspicion detection (not mere topic
  mention), degeneracy detection + early termination (agents were mirroring each
  other), per-(conversation, turn, speaker) seed derivation (replicates were not
  independent), and a `parse_failed` guard (the fallback parser was feeding the
  JSON *template placeholder* to the opponent).
- **The detector reality check.** The regex that fires `t_think` was measured
  against 231 hand-labelled sentences: **precision 0.185** against a 0.8 gate. A
  bag-of-words baseline matched it, while the annotator ceiling was 0.986 — i.e.
  *the task is genuinely not lexical*, and no pattern-matcher will do. A semantic
  stance classifier is required. **→ Pushed:** `t_think` is a known-broken
  instrument to be replaced; it blocks *interpretation*, not *collection*.
- **Gate 2 (conversations don't collapse):** passed at n=20 (15% degenerate).
  But **censoring was ~90%** — agents form private suspicion and essentially
  never publicly commit within 12 turns. **→ Pushed:** survival analysis is
  mandatory, and the censoring rate is itself a headline ("privately suspicious,
  publicly silent").

The bottleneck was never the instrument, though. It was that **the fine-tune had
never run.** So we went to GPU.

### 3.1 Stage 0 — first GPU hours (2026-07-28)

Validate the pipeline end to end on free hardware and answer one question: *does
a fine-tuned adapter still emit `<think>` blocks?*

- Kaggle T4, pilot corpus, 30 optimizer steps, 78 min, $0. Loss 1.43 → 0.80,
  trainable params exactly 1.0491% (rank-32 arithmetic checks out — embeddings
  untouched). **PASS** — the format survived at this small dose.
- Live hardware shook out four real bugs: `torch.cuda.is_bf16_supported()`
  returns True on a T4 via software emulation (switched to compute-capability);
  the chat template *pre-opens* `<think>` so completions carry only `</think>`
  (extractor fixed); the think-block verifier was testing open-ended riddles
  instead of the real task; and `probe_eval.py` was scoring the *think block*
  instead of the answer. Measured T4 throughput: **157.4 s/step**.
- **→ Pushed:** the pipeline works; but 30 steps / 311K tokens cannot shift a
  reasoning prior. Do a properly-dosed run (full canon).

### 3.2 The format collapse (2026-08-06) — the pivotal finding

Controlled measurement, n=8 per arm, one base model in VRAM with adapters
swapped by name (identical weights/prompts/sampling):

| arm | steps | unique tokens | **closure** |
|---|---|---|---|
| base, no adapter | — | — | **8/8** |
| stage-0 | 30 | 311K | **8/8** |
| full canon | 103 | 3.36M | **1/8** |

Fine-tuning on the full canon **destroyed** the think-block format that survived
intact at 30 steps. Ruled out by design: not the model/prompt/budget (base is
8/8 on identical inputs), not truncation (7 of 8 failures stopped naturally),
not the extractor, not dtype. The mechanism is **catastrophic forgetting of the
RL-trained reasoning format under raw-text causal LM** — every gradient step
optimises "predict the next word of Victorian prose", none reinforce "emit and
close `<think>`".

**Why this outranked a bug:** the premise requires a dose large enough to shift a
reasoning prior (~1M+ unique tokens, per LIMA / Betley et al.). **The dose that
produces an effect may be the same dose that destroys the channel the effect is
measured through.** If so the design cannot work as specified. **→ Pushed:** the
14B spend was BLOCKED; map the dose curve before spending anything; and separate
the confound (stage-0 and full-canon differ on *both* steps and unique tokens).

### 3.3 Mapping the dose curve (2026-08-07, 2026-08-08)

Re-run the full canon keeping **every** checkpoint, then score closure at each —
the checkpoints *are* the dose-response experiment. Two independent runs
(train_loss bit-identical at `0.848246…`, so the checkpoints are deterministic):

| run | early (≤35 steps) | late (≥45 steps) | Fisher p |
|---|---|---|---|
| 2026-08-07 (12 ckpts) | 0.84 [0.68, 0.93] | 0.50 [0.37, 0.63] | 0.0014 |
| 2026-08-08 (22 ckpts) | 0.70 [0.57, 0.80] | 0.42 [0.33, 0.52] | 0.0015 |

Degradation is real and significant, and **begins early** — closure is already
sliding by step ~20 (~655K tokens, *below* the effect threshold). Crucially, the
per-point numbers wobble run-to-run (step-25 went 6/8 → 4/8) while the pooled
contrast holds to three decimals. **→ Pushed:** we retracted the "collapse at
step N / usable window at ~35" framing — n=8 per point cannot certify any single
step as safe — and replaced it with Wilson intervals + a pooled early-vs-late
Fisher test. The supportable claim is *"degradation begins at or below the
effect threshold."*

### 3.4 The durability disaster (and the fix)

Four separate runs were lost to Kaggle wiping `/kaggle/working` — first
`save_total_limit=3` silently discarding checkpoints, then session-end wipes,
then a run that trained 5 h with **no HF token attached** so nothing persisted.
Each loss cost hours. Fixes, escalating:

1. Persistence **built into the code** — `train_lora.py` uploads every checkpoint
   to HuggingFace the instant it is written; uploads retry with backoff and never
   crash the run.
2. When that was still skippable (a missed warning), we made it **fail-closed** —
   training now *aborts before it starts* if a repo is configured and no token is
   reachable.
3. Lost *measurements* were reconstructed faithfully from the runs' stdout
   (analysis recomputed by the real code), with provenance stamped.

**This is itself a genuine contribution:** "an experiment you cannot read
mid-flight is gambling," enforced in code rather than left to operator
diligence. Once persistence was fail-closed, every subsequent run survived.

### 3.5 Confound resolved: steps, not breadth (2026-08-12)

The collapse still confounded two knobs. We isolated them with **one** run —
the pilot corpus (311K unique) trained to ~110 steps by re-reading it 11×, on a
matched cosine schedule, so the *only* difference from the full-canon curve is
corpus breadth (1/11th). Filling the last cell of a 2×2:

- **At matched low dose, pilot ≈ canon** (0.70 vs 0.70, Fisher p = 1.0) despite
  11× the unique tokens → breadth does nothing.
- **The pilot collapses with steps at 1/11 the breadth** (0.70 → 0.21,
  p = 2.7e-09).
- **The low-breadth pilot is *worse* at high steps** than the broad canon
  (0.21 vs 0.42, p = 0.0012) — re-reading a narrow corpus 11× damages the format
  *more* than one pass over a large diverse one.

**Verdict: optimizer STEPS / cumulative weight movement drive the collapse, NOT
unique-token breadth.** **→ Pushed:** the lever is a *constrained subspace* (low
LoRA rank), not a smaller or curated corpus. Try rank 8.

### 3.6 Low-rank mitigation: the format is rescued (2026-08-14)

Rank 8 (vs 32), everything else identical, so the closure curves overlay
directly. Measured as a **dual** run — closure *and* held-out perplexity — because
low rank could "preserve closure" by simply learning too little to matter:

- **Closure far better preserved:** early **0.96** / late **0.73** (vs r32's
  0.70 / 0.42). Sweet spot at step ~50: closure **8/8**.
- **Held-out Speckled Band perplexity dropped +43.8%** (H1 gate ≥ 5%).
- Automated verdict: **RESCUED** — a window with intact format *and* effect.

For the first time in the project, a fine-tuned adapter kept the reasoning format
*and* showed a large held-out effect. This cleared the blocker. **→ Pushed:**
verify the effect is real (perplexity is a proxy) before declaring victory.

### 3.7 The H2 correction: the effect is mostly generic (2026-08-14)

The run computed WikiText perplexity (the "did general language survive"
guardrail) but the driver didn't print it. We pulled it:

| step | Holmes drop | **WikiText drop** | Holmes-specific excess | Holmes/Wiki ratio |
|---|---|---|---|---|
| base | — | — | — | 1.122 |
| 50 | +42.5% | **+34.4%** | +8.1% | 0.984 |
| 103 | +43.8% | **+34.1%** | +9.7% | 0.956 |

**WikiText perplexity dropped ~34% too.** The model didn't forget English — it
got much *better* at predicting *all* prose. The reason: base R1-Distill is a
reasoning model, genuinely poor at raw prose next-token prediction, so training
on *any* prose restores prose-LM and drops perplexity on everything. So **~34 of
the 44 Holmes points are generic recovery; only ~10 are Holmes-specific** (the
Holmes/WikiText ratio moving 1.12 → 0.96). **→ Pushed:** perplexity is confirmed
inadequate as the effect measure. The behavioural check is now *required*, not
optional. (Had we reported "+44% = learned Holmes," we would have been wrong —
the guardrail caught it.)

### 3.8 The behavioural check: no reasoning shift (2026-08-14)

We ran the base model and the step-50 adapter on the same 30 probe prompts
(10 deduction-inviting, 10 reasoning, 10 neutral control), greedy-decoded, and
**read the actual `<think>` blocks side by side.**

- Where both models produce a block, the reasoning is **strikingly similar** —
  both walk cue-by-cue, hedge heavily, and neither reaches a confident deduction.
  On the classic "tanned hands, pale face, calluses, checking a watch" prompt,
  base ends *"I'll let him be,"* fine-tuned ends *"I can't be sure without
  asking"* — the fine-tuned trace drifts *toward* concern/helping, away from the
  detective register.
- The register-marker table's apparent "hedging drop" on deduction prompts was
  partly an **artifact**: on two prompts the fine-tuned model produced *no think
  block at all*, and absence scored as zero hedging.

**Verdict: no visible reasoning shift.** Behaviourally this is the mitigation's
TOO_WEAK branch — perplexity said RESCUED, the transcripts say null. The Phase-1
precondition (a *distinguishable reasoning prior*) is **not met** at step-50.

---

## 4. Consolidated results

| finding | evidence | status |
|---|---|---|
| Pipeline runs; format survives at tiny dose | Stage 0, 8/8 closure | verified |
| QLoRA on raw prose destroys the `<think>` format | n=8 controlled, 8/8→1/8 | verified, replicated |
| Collapse driven by **steps / weight movement**, not breadth | 2×2, p=2.7e-09 vs p=1.0 | verified |
| **Low rank (r8) preserves the format** | closure 0.96/0.73 vs 0.70/0.42 | verified |
| Perplexity effect is **mostly generic** prose recovery | WikiText −34% vs Holmes −44% | verified |
| **No reasoning shift** from prose fine-tuning | paired think-block transcripts | verified |
| `t_think` regex detector not viable | precision 0.185 vs 0.8 gate | verified |
| Conversations ~90% censored (never publicly commit) | Gate 2, n=20 | verified |
| Commitment gap itself | — | **never measured** (Phase 2 unreached) |

**Total GPU spend: ≈ $0.** Every result above is free.

---

## 5. The unifying diagnosis

Every problem traces to **one root cause: raw prose is the wrong training
signal.** It fails in two independent ways at once:

1. **It never reinforces the think format** → the format collapses (§3.2–3.6).
2. **It contains no reasoning** → even when the format survives, the reasoning
   doesn't shift (§3.8).

Continued-pretraining on Watson *narrating* Holmes's deductions optimises
"predict Victorian detective prose." Nothing in that objective says "reason
deductively inside your own think block." So the model learned the *prose*
(perplexity dropped) and left its *reasoning* untouched. **You trained on
descriptions of deduction and hoped to shift the model's own deduction — a
channel mismatch.**

---

## 6. Where we are now

- The **format blocker is genuinely solved** (low rank).
- The **behavioural manipulation is not** — prose fine-tuning does not produce a
  distinguishable reasoning prior.
- The **conversation arm stays blocked** — two base-equivalent reasoners would
  measure noise, not a commitment gap.
- The project's **centre of gravity has shifted** from "measure the commitment
  gap" (unreached) to a **methods finding about fine-tuning distilled reasoning
  models** — which is smaller than the original ambition but *true and verified*,
  which the original would not yet have been.

---

## 7. Next steps (the pivot)

The mechanical answer to "what training would actually force the model to think
that way?" follows directly from §5: **train on the reasoning itself, in the
think channel, with the loss on the reasoning** — i.e. **supervised fine-tuning
on deductive reasoning traces (reasoning distillation)**, not continued-
pretraining on prose. Each example is `prompt → <think>observation → inference →
confident conclusion</think> answer`, trained on the completion. This fixes
*both* failure modes: the data *contains* `<think>…</think>`, so training
*reinforces* the format instead of destroying it; and the loss is on the
reasoning, so the model learns to reason that way. Per LIMA, ~1,000 excellent
examples suffice — not 3M tokens of prose.

**Open decisions, in order:**

1. **Cheap viability probe first (no training):** can the *base* model,
   few-shot-prompted with real canon deductions as exemplars, produce a
   confident deductive think block on a held-out prompt? If yes,
   **self-distillation** has clean source traces (respects the standing rule that
   generated blocks come from the base model itself). If no, we face a
   stronger-model-generator decision (new provenance question).
2. **Contamination boundary:** train on *general* observation→inference deduction
   (strangers, objects, scenes), **not** on AI-detection — so the commitment gap
   in Phase 2 stays an emergent measurement, not something trained in.
3. **Learned vs prompted:** confirm the reasoning prior must be *in the weights*
   (SFT). If system-prompt conditioning would satisfy Phase 2, the entire
   Phase-1 training problem dissolves — but it is not "learned," and not the
   contribution.
4. After SFT: **re-run the exact thinking-shift check we already built** to
   verify a real reasoning shift on *held-out* prompts (generalisation, not
   memorisation), then — only if it holds — proceed to Phase 2.

**Explicitly deferred / shelved:** step-103 closure check (low expected payoff),
the 14B/32B model upgrades (no point until Phase 1 works), and the rehearsal
sketch as originally framed (subsumed by the SFT-on-traces plan).

---

## 8. Process lessons worth keeping

- **Persist off-machine as artifacts appear, enforced in code and fail-closed.**
  Four lost runs taught this; the fix (per-checkpoint upload + refuse-to-start
  without a token) is a reusable pattern.
- **Reconstruct measurements from stdout when weights are lost.** A labelled
  reconstruction that reproduces the printed statistics beats discarding a real
  result over file-identity purity.
- **A proxy passing spectacularly is not the real path.** Perplexity "RESCUED"
  would have been reported as success; the guardrail (H2) and then the
  transcripts told the truth. Verify the real artifact via the real channel.
- **n=8 per point cannot certify a boundary** — report intervals and pooled
  contrasts, not a single "collapse step."

---

## 9. Budget

$50 cap. Spent ≈ **$0** — all training on free Kaggle T4, all inference/probing
local or free-tier. The staged 14B (~$2) and 32B (~$6) spends were never
incurred, because Phase 1 never cleared the gate that would justify them.
```

---
*Related detail: `format_collapse_20260806.md`, `dose_curve_20260808_writeup.md`,
`confound_20260812_writeup.md`, `mitigation_lowrank_20260814_writeup.md`,
`thinking_shift_20260814_writeup.md`, and the full Decision Log in `CLAUDE.md`.*
