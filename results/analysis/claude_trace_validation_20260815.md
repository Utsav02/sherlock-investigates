# Claude-distilled traces — validation run (2026-08-15)

De-risk step for the STaR/RFT pivot (Decision Log 2026-08-15). **No GPU, no SFT.**
First real execution of Claude trace generation.

```bash
python scripts/data_prep/generate_traces.py --backend claude \
    --scenarios data/sft/scenarios_seed_claude.jsonl --samples 1 \
    --out data/sft/traces_claude_validation.jsonl
```

Teacher: Claude Code CLI 2.1.229 (headless `-p`), reported `modelUsage` =
`claude-sonnet-5`. 18 usable scenarios × 1 sample, ~3.8 min wall-clock.

---

## Verdict: the trace source is VALIDATED. Proceed to scale.

Both gates pass. The reasoning is genuine deduction, not fluent filler.

| axis | base r1:7b (`traces_demo.jsonl`) | Claude | gate |
|---|---|---|---|
| keeper rate (ground-truth filter) | **0/10** | **13/18 (72%)** | pass |
| think block present | 10/10 | **18/18** | pass |
| raw output opens with `<think>` | n/a | **18/18** | pass |
| answer opens with "This is" | n/a | **18/18** | pass |
| form-vs-substance (traces READ) | commits to confident-**wrong** generic answers | **genuine cue→inference→identity** | pass |

Format compliance was perfect on the **first** prompt attempt — no iteration was
needed, so nothing here is tuned to its own test set.

---

## Axis (a) — keeper rate: 13/18, and the 5 misses are NOT reasoning failures

Every miss was read. None is a case of the teacher failing to deduce.

| # | ground truth | Claude's answer | reading |
|---|---|---|---|
| 0 | retired sergeant, **Royal Marines** | "a former soldier turned commissionaire … foreign-service tattoo … drilled reflexes of his old regiment" | **filter-lossy.** Substantively right (retired serviceman); the cues never encode *which* service, so "Royal Marines" is unrecoverable in principle. |
| 13 | recently **emigrated** from a hot country to a cold one | "newly **returned** from months of outdoor manual labor in a hot foreign country, so recently back that she has not yet readjusted to the cold" | **filter-lossy.** The deduction is correct; only the emigrate/return word choice differs. |
| 11 | a gambler on a long losing streak | "respectable standing who has recently fallen into financial difficulty … putting on a brave face" | **under-determined scenario.** Ruin is deduced; the *cause* (gambling) is not in the cues. |
| 14 | a church organist | "a professional drummer … hi-hat, snare, bass pedal" | **under-determined scenario.** Limb independence + heel/toe pedal wear + finger splay fit organist *and* drummer. |
| 17 | bus/tram conductor, end of a double shift | "a delivery or collection roundsman … fixed route … change until counting is automatic" | **under-determined scenario.** Fixed route + satchel + automatic change-making fit both. |

**So the binding constraint is the filter and the scenarios, not the teacher.**
2 of 5 misses are correct deductions the keyword matcher cannot see; 3 are
scenarios whose cues admit more than one valid answer. Confirmed directly:

```
detect_leak("a retired sergeant of the Royal Marines", [],
            "This is a former soldier turned commissionaire…")  ->  (False, [])
```

This is exactly the contingency the Decision Log flagged — *"the keyword match is
a first cut; upgrade to an LLM-judge verifier (V-STaR style) if it proves too
lossy on semantic answers."* It has proved too lossy. **True keeper rate is
~15/18 on substance vs 13/18 as scored.** Recommended before scaling: an
LLM-judge verifier, and drop/repair the 3 under-determined scenarios (a cue set
that admits two answers is a defective training item regardless of verifier).

## Axis (b) — form vs substance: the load-bearing gate

Five traces read in full (ids 3, 4, 9, 15, 14). They take **each cue in turn,
give a physical mechanism for it, rule out alternatives, and chain to one
conclusion** — the target shape, with real content.

> **[9] farmhand.** "The burn is confined to the back of the neck and the
> forearms, while the forehead stays pale. That pattern is the signature of a hat
> brim and a bent posture… A man merely walking about in summer would burn across
> the whole face." Then chaff → "husk from threshed grain… near a rick or
> threshing floor, not a city street"; split palms → "gripping a rough wooden
> shaft… a scythe, a fork, or plough handles — not a desk, not a shop counter."

That is causal reasoning about *why* each mark exists, plus explicit negative
evidence. Compare the base model, which produced fluent think blocks and then
answered "clerk or copyist" to almost every scenario regardless of cue.

Quantified over all 18 traces:

| check | result | reading |
|---|---|---|
| pairwise Jaccard vocab overlap | mean **0.062**, max 0.127 | traces are lexically distinct — not one template reskinned |
| hedge words (`maybe/might/could be/perhaps`) | **1** across 18 traces | the base model's hedging failure mode is absent |
| answer copied verbatim into `think` | **0/18** | the conclusion is derived, not restated |
| think length | 713–1801 chars (mean 1109) | matches the ~1293-char real-task think block measured on the student (2026-07-28) — right size for SFT |
| "taken together / combine them" closer | 8/18 | mild structural regularity, see caveat |

**The strongest evidence against form-only mimicry is miss [14].** The teacher
reasoned from limb independence and heel-toe pedal wear to *drummer* — a
defensible answer the scenario's author did not intend. A model rationalising
backwards from a known label cannot land somewhere the label isn't. The teacher
is genuinely reasoning forward from the cues.

### Caveats that travel with this result

1. **n=18, one sample each, one teacher.** Enough to validate a source, not to
   characterise it. Re-read a sample of every future batch.
2. **8/18 share a "taken together…" closing move.** It is the chaining step the
   prompt asks for, and it is under half the set, so it is not templating in the
   damaging sense — but if it hardens at n=200 the student may learn the closer
   as a tic. Worth re-measuring at scale.
3. **This validates the traces, NOT the student.** Everything here is teacher
   output. Whether SFT transfers substance or only form is settled by the
   thinking-shift held-out audit *after* training — the second prescribed
   guardrail, still outstanding and still mandatory.
4. **Scenario quality is now the weak link**, not trace quality. 3/18 seeds are
   ambiguous. Scaling scenario generation without a disambiguation check will
   propagate that rate.

---

## Implementation notes

* `--backend {ollama,claude}` on `generate_traces.py`; provenance stamped per row
  (`generator`), raw teacher output retained in `raw` for audit. `seed` is
  recorded `null` for the Claude backend — the CLI has no seed parameter, so
  recording the requested value would have implied a reproducibility we lack.
* **`claude_chat` had never been executed and was broken.** It looked up `claude`
  on PATH only; the CLI is *not* on PATH on this machine, so every
  `--backend claude` call would have raised `FileNotFoundError`. Replaced with
  `resolve_claude_bin()` (CLAUDE_BIN → PATH → desktop-app bundle), mirroring the
  working resolver in the French project. The binary here resolves to
  `~/Library/Application Support/Claude/claude-code/2.1.229/…/claude`.
* **Calls run in an empty temp cwd.** Invoked inside the repo, the CLI loads this
  project's large `CLAUDE.md` as context: a trivial probe cost **$0.34** in-repo
  vs **$0.072** from a clean directory (measured via `--output-format json`), a
  ~4.7× saving per call for context that is irrelevant to trace generation. The
  18-call run itself was not individually metered.
* `ANTHROPIC_API_KEY` / `ANTHROPIC_AUTH_TOKEN` are dropped from the subprocess
  env so the CLI uses subscription auth rather than silently billing a key that
  happens to be in the environment (same guard as the French project).
* 7 new parser/prompt tests; suite **106 tests, OK**.
