# Data strategy for the SFT-on-reasoning-traces pivot

**Written 2026-08-14.** Forward-looking planning doc (not a result). Companion to
`results/analysis/experiment_journey_20260814.md`. Decides what data the Phase-1
retrain should use, having established that continued-pretraining on the Holmes
prose corpus fails (wrong channel — see the journey doc).

---

## 1. What the SFT actually requires

Each training example must be:

1. **Right channel & format** — `prompt → <think>first-person step-by-step
   reasoning</think> answer`, with the think block *present in the target*. This
   is what reinforces the format (fixing the collapse) instead of destroying it.
2. **Right reasoning type** — *abductive/forensic*: observe cues → infer a hidden
   fact (identity, history, state) → **commit** to a confident, specific
   conclusion. Not formal symbolic logic; not hedged enumeration.
3. **Right scale & quality** — ~1k–few-k *diverse, confident, valid* traces
   (LIMA / Betley range), **curated** (raw generations need filtering).
4. **Clean provenance** — permissively licensed or model-generated, and **not**
   about AI-detection / hidden-role detection, so the commitment gap stays
   *emergent* rather than trained-in.

## 2. How SFT handles "good vs bad" (and why curation is the whole game)

**Plain SFT is imitation.** The loss is next-token cross-entropy on the target
completion; the model is pushed toward reproducing those exact tokens. It has
**no notion of "bad"** — a hedgy trace in the data teaches hedging. So quality is
controlled entirely by **what you put in the set**:

| method | needs | teaches good-vs-bad? | fit for us |
|---|---|---|---|
| **SFT** (imitation) | good examples only | No — curation is the only lever | primary |
| **Rejection-sampling SFT (RFT)** | generate → filter → SFT survivors | implicitly (the filter) | primary (how we curate) |
| **DPO** | `(prompt, chosen, rejected)` pairs | **yes**, directly, no reward model | optional sharpener |
| **RLHF / PPO / GRPO** | reward model / verifier | yes (graded) | overkill / over-budget |

**DPO is unusually well-suited to our exact failure.** The diagnosed problem is
"hedge & enumerate instead of commit." A DPO pair on the *same* prompt — `chosen`
= a confident Holmes trace, `rejected` = the base model's own hedgy trace —
isolates precisely that, and we **already have the rejected examples** (base-model
outputs from the viability probe and thinking_shift).

**Experiment-integrity caveat.** Our dependent variable *is* a commitment
behaviour. Pure SFT on good traces makes confidence a *byproduct* of the reasoning
style (cleaner, more emergent); DPO optimizing "commit vs hedge" makes decisiveness
a *directly optimized target* (more powerful, but the commitment-gap result then
needs extra scrutiny for being partly baked in). Keep all training on *general*
forensic deduction; never on AI-detection. Recommendation: **SFT/RFT primary, DPO
optional and reported transparently.** (Precedent: R1-Distill itself got its
reasoning via SFT on R1 traces.)

## 3. Critique of the Sherlock corpus

| requirement | Sherlock canon |
|---|---|
| Right format (think-block traces) | ✗ narrative prose (Watson narrating) — the root cause |
| Right reasoning type (forensic abduction) | ✓✓ perfectly matched |
| Scale of actual deductions | ✗ a few dozen set-pieces across 60 stories |
| Provenance / license | ✓ public domain; ✓ deliberately *oblique* to the measured task |

**The problem was never the domain — it was the format.** And Sherlock's domain
has a subtle virtue: forensic abduction about *strangers and objects* is the right
reasoning **style** while being **one step removed** from the measured **task**
(infer human-vs-AI from conversation). That obliqueness *protects the
measurement*. So keep Holmes as the **style / exemplar / domain anchor**; stop
using the canon *as the training set*.

## 4. Landscape of external sources, by role

No existing dataset is drop-in (right-format **and** right-domain **and**
cleanly-licensed). They are *ingredients*:

| source | what it gives | role | license (verified where noted) |
|---|---|---|---|
| **OpenThoughts-114k / OpenR1** | the exact `<think>` format, high quality | **format anchor / rehearsal** — mix a slice in so SFT reinforces the channel | **Apache-2.0 ✅** (verified; traces from R1/MIT) |
| **αNLI / ART** (allenai/art) | abductive *observation → hypothesis* pairs at scale (~20k/200k) | **seed scenarios** for generation (diversity the canon lacks) | **⚠️ uncertain** — repo Apache-2.0 but data derived from **ROCStories** (form-gated, terms unstated) |
| **Social-deduction sets** (Avalon-NLU, Werewolf; CSP4SDG, GRAIL, InMind) | hidden-identity-from-deception — the *actual task flavor* | **eval + framing only** (see §6) | research data — ⚠️ verify; contamination risk if trained |
| **DetectiveQA, TurnaboutLLM, WhoDunIt** | detective deduction *with* reference reasoning chains | **eval only** | ❌ derived from copyrighted novels/games |
| **FOLIO, ProofWriter, LogiQA, ReClor** | formal deductive chains (proofs / FOL) | **control-variant corpus** (a different, distinguishable prior) | mostly permissive/synthetic — verify each |
| **Sherlock canon** | forensic-abduction style + exemplars | **exemplar bank / style anchor** | Public domain ✅ |

## 5. License verification (done 2026-08-14)

- **OpenThoughts-114k — Apache 2.0.** Commercial use + model release OK. Safe to
  train on and ship. ✅
- **ART / αNLI — unclear.** HF lists license "unknown"; the AllenAI code repo is
  Apache-2.0, but the *data* comes from **ROCStories**, which is "free to
  everyone" yet **gated behind an access form** with **no stated commercial or
  redistribution terms**. → Fine for private research; **do NOT redistribute ART
  text or rely on it for a released adapter without confirming terms.** Safer:
  use ART only to *shape our own generated scenarios* (the shippable data is then
  model-generated on self-authored prompts), or confirm ROCStories terms first.
- **Sherlock canon — public domain** (all Holmes entered US public domain 2023).
  Safe. ✅
- **Detective benchmarks (DetectiveQA / TurnaboutLLM / WhoDunIt)** — derived from
  copyrighted novels and games. **Eval only; not training data for a released
  model.**
- **Formal-logic sets (FOLIO / ProofWriter / LogiQA / ReClor)** — generally
  permissive or synthetic, but **verify each** before use.

## 6. The reframe worth holding onto

The *closest* domain match to the real experiment is not detective fiction — it's
**social deduction** (infer a hidden identity from deceptive conversation:
Werewolf, Avalon). That is almost exactly "spot the AI." **But that closeness makes
it dangerous as training data** — train on hidden-role-detection reasoning and you
contaminate the exact suspicion channel you measure. So social-deduction data
belongs in **evaluation and framing**, not the training signal — which is
*precisely why the oblique Holmes proxy is the right training target*. The critique
resolves in Sherlock's favour on domain; it only ever failed on format.

## 7. Recommended data strategy (recompose, don't replace)

1. **Demote the canon** to a few-shot exemplar bank + style reference (already how
   the viability probe uses it).
2. **Generate the training traces** (self-distillation) — no clean off-the-shelf
   set exists; confirmed by §4.
3. **Seed generation from diverse abductive scenarios.** Use ART's *structure*
   (observation → infer explanation) for diversity/scale — but, given the license
   uncertainty (§5), prefer generating our *own* scenarios in that shape (or
   confirm ROCStories terms) so the shippable data stays clean.
4. **Mix ~10–20% OpenThoughts R1 traces as a format anchor** during SFT — an
   off-the-shelf, Apache-2.0 "rehearsal" that reinforces the `<think>` channel and
   directly counters the collapse.
5. **Curate hard (rejection sampling):** several traces per seed, keep only the
   confident-deductive ones. Optionally **DPO** (confident vs the base model's
   hedgy trace) to sharpen commitment — cautiously (§2).
6. **Reserve the formal-logic sets as a *control* corpus** — a second,
   cleanly-distinguishable reasoning prior (the experiment wants multiple priors,
   not just Holmes-vs-base).
7. **Use social-deduction + detective benchmarks as held-out *evaluation*** of
   whether deductive reasoning improved — never as training.

## 8. Decision (2026-08-14) — self-source via reverse construction

The license gate is **resolved: self-source the scenarios; ART/ROCStories dropped
from the critical path.** Rationale (see Decision Log 2026-08-14): confirming
ROCStories terms is slower, uncertain (research datasets are often
non-commercial/no-redistribution), and — decisively — ART's *content* is mundane
everyday-story abduction, not forensic profiling, so it is a weak fit even if the
licence clears. Self-sourcing is easier, unconditionally feasible, and a *better
content fit*, via **reverse construction**:

> Start from a KNOWN answer (identity/occupation/situation) → have the model
> invent the observable CUES that imply it → the cues become the scenario, the
> known answer is the ground truth.

This gives forensic scenarios, **crisp answers by construction** (removing the
ambiguity that made the base model hedge in the viability probe), and a
**ground-truth target to filter traces against** (automatic rejection sampling).
Built as `scripts/data_prep/reverse_scenarios.py` (local Ollama, self-contained).
OpenThoughts (Apache-2.0) stays as the format anchor. ART is optional future
diversity only, never a blocker.

## 9. Build order (unblocked)

1. **`scripts/data_prep/reverse_scenarios.py`** — reverse-construct seed scenarios
   (identity → cues → scenario + ground truth). ✅ built, with an automated
   **leak filter** (`detect_leak`, the curation gate) and two backends.
   **Generator finding (2026-08-15):** the local base 7B is too weak — a 30-seed
   run gave **7/30 usable** (`data/sft/reverse_scenarios_seed42.jsonl`) with
   answer-leaks, code-switching, and nonsense cues. Fix: **`--backend claude`**
   (headless Claude Code CLI, like other projects call it), which is
   provenance-safe because scenarios are *prompts*, not distilled traces. A
   Claude-generated seed set is **18/18 usable**
   (`data/sft/scenarios_seed_claude.jsonl`). Determinism is a non-issue: the
   committed JSONL *is* the reproducible artifact (generate once, version it,
   stamp the generator); LLM sampling need not be deterministic.
2. **Trace generation** — for each `scenario_prompt`, generate a Holmes-style
   deductive `<think>` trace (base model, few-shot; the viability probe's harder
   commit-or-nothing instruction).
3. **Filter (rejection sampling)** — keep only traces whose conclusion matches
   `ground_truth`. This is the quality signal (§2: SFT has no other one).
4. **SFT** on survivors, mixing ~10–20% OpenThoughts R1 traces as a format anchor.
   Optional **DPO** (confident keeper vs the base model's hedgy trace) to sharpen
   commitment — cautiously (§2).
5. **Verify** — re-run the existing `thinking_shift` check on held-out prompts
   (generalisation, not memorisation), then — only if it holds — Phase 2.

---

**Sources:** αNLI/ART (arXiv 1908.05739; huggingface.co/datasets/allenai/art;
github.com/allenai/abductive-commonsense-reasoning); ROCStories
(cs.rochester.edu/nlp/rocstories); OpenThoughts (arXiv 2506.04178;
huggingface.co/datasets/open-thoughts/OpenThoughts-114k); DetectiveQA (arXiv
2409.02465); TurnaboutLLM (arXiv 2505.15712); WhoDunIt (arXiv 2502.07747); FOLIO
(arXiv 2209.00840); LogiQA 2.0; Avalon-NLU (sstepput.github.io/Avalon-NLU);
Bayesian Social Deduction / GRAIL (arXiv 2506.17788); InMind (arXiv 2508.16072).
