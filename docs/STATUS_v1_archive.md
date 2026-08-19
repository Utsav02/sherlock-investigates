# ARCHIVED — V1 status (frozen 2026-08-15)

**This file is history, not instructions.** It was `STATUS.md` until 2026-08-18,
when the active project became v2 and this content was archived verbatim.

**Do not resume from this file.** Its "NEXT SESSION" block tells you to scale
Claude scenario generation to ~300 and assemble the V1 SFT set. That work is
**paused**: `v2/experiment_design.md` §10 defers D1–D5, and D5 is not authorised
by the design at all. The live status is `../STATUS.md`.

Kept because the V1 measurements, decisions and retractions in it are real and
still cited by the Decision Log.

---

# STATUS — sherlock-investigates
Updated: 2026-08-15 (judge keeper-scan CLEAR for pilot — cleared to scale)

## DONE (session 3) — PILOT keeper scan: 0 gross false accepts
Writeup: `results/analysis/keeper_scan_20260815_231156.md`; Decision Log
2026-08-15 (fourth entry). Presenter: `scripts/eval/build_keeper_scan.py`
(read-only) → `results/analysis/keeper_scan.html`.

- **14/14 keepers correct, 0 gross false accepts.** Scanned only the KEEPERS,
  because for a pilot the only judge error that damages training DATA is a false
  accept; false rejects cost yield, which the 2× plan absorbs.
- **⚠️ This was a CLAUDE-ON-CLAUDE cross-check, NOT an independent audit.**
  Teacher, judge and scanner are one model family — a **shared blind spot is not
  excluded**. It catches gross false accepts only.
- **The rigorous ~30-label precision/recall audit is DEFERRED to just before the
  final scaled run, and must use a HUMAN or NON-CLAUDE annotator.** Its scope is
  unchanged by this scan.
- 4 coarser-than-seed answers (ids 0/3/5/7) are the rubric working as written
  (coarser permitted, contradictory forbidden). 1 borderline NOTED not dropped:
  id 13, "emigrated" vs "returned" — right on situation, slips emigrant/returnee,
  and already excluded by the ambiguity gate.
- **No rubric change** — 0 false accepts means nothing to tighten, and retuning
  `JUDGE_SYSTEM` against the same 18 rows is fitting the instrument to its own
  test set (`t_think_07` lesson).
- `cues_miss_gt` **dropped** (see below). 129 tests OK.

## DONE THIS SESSION — Claude trace source is VALIDATED
`--backend claude` built, executed, and judged on both axes. Writeup:
`results/analysis/claude_trace_validation_20260815.md`; Decision Log 2026-08-15
(second entry). Data: `data/sft/traces_claude_validation.jsonl` (18 rows).

- **Keeper rate 13/18** vs base self-distillation's 0/10.
- **Format 18/18** — every output opened `<think>`, every answer "This is",
  correct on the FIRST prompt attempt (not tuned to its own test set).
- **Form-vs-substance gate PASSED by reading 5 traces in full** — genuine
  cue→mechanism→identity deductions with explicit negative evidence. Support:
  vocab Jaccard 0.062, 1 hedge word in 18, 0/18 answers copied into `think`.
  Best evidence: on the organist scenario the teacher reasoned to *drummer* —
  forward reasoning, not backward rationalisation from a known label.
- Fixed: `claude_chat` resolved the CLI on PATH only and could never have run
  (binary is not on PATH here) → `resolve_claude_bin()` + clean-cwd calls
  ($0.34 → $0.072 per call). 106 tests OK.

## DONE (session 2) — both instruments built, re-validated, and they CORRECTED the diagnosis
Writeup: `results/analysis/instrument_upgrade_20260815.md`; Decision Log
2026-08-15 (third entry). Artifacts: `data/sft/traces_claude_validation_judged.jsonl`,
`data/sft/scenarios_seed_claude_disambig.jsonl`.

- **LLM judge is the gate** (`--judge` during generation, `--rejudge` to rescore).
  Keepers 13 → **14/18**; judge↔keyword agreement 15/18, 0 unparseable.
  Recovered the 2 predicted near-misses AND **rejected one the keyword accepted**
  (`[10]` "former *professional* boxer" vs ground truth "*amateur* boxer") — the
  lexical filter was wrong in BOTH directions, and a false accept is worse.
- **Disambiguation: 6/18 ambiguous** — and only ONE is from the three the last
  session guessed. Organist and conductor are **clear** (the check names the
  ground truth as best), so those were **teacher errors, not scenario defects**.
  Four never-suspected scenarios are genuinely ambiguous (violinist/violist,
  fisherman/sailor, med-student/any-crammer, watchmaker/jeweler). The by-eye
  diagnosis was 2 false positives + 4 false negatives out of 6.
- **Combined yield 9/18 (50%)** = keeper ∧ usable scenario. **Plan scenario
  generation at ~2× the target SFT count** (~300 scenarios for ~150 examples).
- Two real parser defects fixed and frozen as tests: `VERDICT: TIE`, and
  `VERDICT: CLEAR` + `BEST: NONE`. Resolution: **BEST is authoritative.**
- `cues_miss_gt` check: 12 opportunities, **fired 0 times**. **REMOVED
  2026-08-15** (session 3) — the gambler case it was built for turned out to be
  *ambiguous*, so `best` was NONE and the check never ran on the one scenario it
  was designed to catch; another batch would reproduce that zero for the same
  structural reason. Halves the disambiguation pass's call count. Reinstate from
  git history if a scaled batch shows the defect. 129 tests OK.

## NEXT SESSION (cleared to scale — still no GPU)
1. **Scale scenarios to ~300** (not ~150) given the 50% yield, running the
   disambiguation check as a generation-time gate. The judge's keeper gate was
   cleared for the PILOT by the session-3 scan, so this is no longer blocked.
2. Generate traces with `--backend claude --judge`, then assemble the SFT set
   (+ OpenThoughts format anchor).
3. Re-read a trace sample from every batch; watch whether the "taken together…"
   closer (8/18 in session 1) hardens into a tic at scale.
4. **Before the FINAL scaled run — the rigorous ~30-label judge audit**, with a
   HUMAN or NON-CLAUDE annotator, rejects included, blind. The session-3 scan
   does NOT substitute for this and does not reduce its scope.

Still mandatory and NOT addressed by this session: the **thinking-shift held-out
audit after training**. Everything validated so far is teacher output; whether
SFT transfers substance or only form is unsettled until the student is read.
Keep local/Claude, no GPU. Full plan: `docs/data_strategy.md`.

## Most recent event (2026-08-14) — thinking-shift check came back NULL

## Most recent event (2026-08-14) — thinking-shift check came back NULL
Read the actual base-vs-step-50 `<think>` blocks (transcript pulled from HF,
now in git). **Base and fine-tuned reason the same way** — cue-by-cue, hedged,
no confident deduction, no Holmes voice. Phase-1 precondition (distinguishable
reasoning prior) NOT met at step-50. Behaviourally this is TOO_WEAK despite
perplexity's RESCUED.
- Three signals agree: H2 decomposition (~10pp Holmes-specific), markers
  (flat/generic + a no-think-block artifact inflating the hedging drop), and the
  transcripts (decisive).
- **Cause:** corpus is the Holmes canon (Watson NARRATING deduction, prose) —
  not reasoning transcripts. Trains prose prediction (PPL drops), not the
  model's own private reasoning. Channel mismatch.
- Evidence in git: `thinking_shift_20260814_171042_transcript.md` +
  `thinking_shift_20260814_writeup.md`. Decision Log 2026-08-14 (4th entry).
- **Conversation arm stays BLOCKED** — two base-equivalent reasoners would
  measure noise.

## Fork now
1. Cheap: check **step-103** (highest Holmes-specific excess; closure 7/8) —
   free ~15 min, low expected payoff, closes the "higher dose?" question.
2. **Rehearsal** — base-model-generated think blocks (reasoning traces in the
   right channel) mixed into training; the one rescue with a mechanism for a
   behavioural shift. More involved; contamination caution.
3. **Reframed writeup** — the full arc is a clean methods story (standard rank
   destroys format; low rank preserves it but the format-safe dose yields only
   generic prose recovery, no reasoning shift, because raw prose is the wrong
   channel). Recommend (1) + draft (3) in parallel; (2) if a shift is still wanted.

## Most recent event (2026-08-14) — low-rank mitigation = RESCUED
Rank 8 (vs 32), full canon. **Closure far better preserved** (early 0.96 / late
0.73 vs r32's 0.70 / 0.42) AND **held-out Speckled Band PPL drops +43.8%** (H1
gate ≥5%). Wide window; sweet spot **step ~50: closure 8/8 AND +42.6%**.
Verdict: **RESCUED → rehearsal NOT needed.** Persistence worked end to end.
- Captured in git: `dose_curve_20260814_042400.json`,
  `effect_curve_20260814_065854.json`, `mitigation_lowrank_r8.json`,
  `mitigation_lowrank_20260814_writeup.md`.
- **H2 PULLED (from HF) — effect is mostly GENERIC.** WikiText PPL dropped ~34%
  alongside Holmes's +44%: the base reasoning-model is poor at raw prose, so
  training on any prose restores prose-LM and drops PPL on everything. Only
  ~10pp is Holmes-specific (the excess; Holmes/Wiki ratio 1.122 → 0.956). The
  RESCUED verdict stands FOR CLOSURE (measured directly); the effect half is
  confounded. Genuine effect JSON (with WikiText) now in git.
- **CAVEATS:** (1) perplexity effect is mostly generic prose recovery, ~10pp
  Holmes-specific and distributional — NOT proof the model *reasons* like
  Holmes. (2) final adapter closure 3/8 — use step ~50.
- **Next (behavioural check now REQUIRED, not optional):** run the behavioural
  effect on step-50 (think-block inspection on deduction prompts) — perplexity
  is confirmed inadequate. If a real reasoning shift shows, the conversation arm
  is unblocked on step-50; if null, points back to rehearsal or a reframed
  writeup. Decision Log 2026-08-14 (three entries).

## NEXT (owner-triggered, free ~15-25 min, NO training): thinking-shift check
`notebooks/kaggle_t4_thinking_shift.py` + `scripts/eval/thinking_shift.py`.
Runs the probe set (10 deduction + 10 reasoning + 10 neutral) through BASE and
the low-rank **step-50** adapter on identical prompts, GREEDY-decoded, and writes
a side-by-side `<think>`-block transcript + a descriptive register profile by
category. NEUTRAL is the control (should move less than deduction prompts).
- **The transcript markdown is the deliverable — READ IT.** Marker numbers are
  descriptive only (task is not lexical; markers saturate R1 traces).
- This is the Phase-1 precondition (did fine-tuning shift the *reasoning*?), NOT
  the Phase-2 headline (commitment gap in conversations). Keep that distinction.
- Adapter pulled from HF (needs HF_TOKEN); CELL 3 checks out the feature branch
  explicitly (fixes the clone-of-main FileNotFoundError). `make test` green (90).
- If a real shift shows on deduction prompts → conversation arm unblocked on
  step-50. If null → rehearsal or reframed writeup. Decision Log 2026-08-14.

## Most recent event (2026-08-12) — confound separator ran, verdict = STEPS
The pilot@110 run completed and **persistence worked end to end** — every
checkpoint uploaded to HF as written (`[persist] checkpoint-5 -> ... (9s)`),
results too; nothing lost. The fail-closed gate + attached token did their job.
- **Verdict: optimizer STEPS / weight movement, NOT unique-token breadth.**
  At matched low dose pilot==canon (0.70 vs 0.70, p=1.0); pilot collapses with
  steps at 1/11th the breadth (0.70→0.21, p=2.7e-09); and pilot is *worse* than
  canon at high steps (0.21 vs 0.42, p=0.0012) — re-reading a narrow corpus is
  MORE destructive than diverse tokens.
- Captured in git: `results/analysis/dose_curve_20260812_104622.json`,
  `confound_pilot103_vs_fullcanon.json`, `confound_20260812_writeup.md`.
- **Next fork:** (1) one cheap low-RANK mitigation run (constrain the subspace,
  protect base weights) — the lever the verdict points to; (2) rehearsal
  (robust fallback); (3) negative-results writeup (solid regardless). Recommend
  (1) + start (3) in parallel. Decision Log 2026-08-12.

## NEXT (owner-triggered, free ~5.5h): low-rank mitigation — the DUAL run
`notebooks/kaggle_t4_lowrank.py` + `configs/kaggle_t4_lowrank_r8.yaml`. Rank 8
(vs 32), full canon, one variable changed so its closure curve overlays the r32
curve. Measures BOTH per checkpoint:
- **(a) closure** — `dose_curve.py` (does the `<think>` format survive?)
- **(b) effect** — `effect_curve.py`, held-out Speckled Band perplexity drop vs
  base (did it learn Holmes?)
`mitigation_analysis.py` overlays them → **RESCUED** (closure ≥0.75 AND PPL drop
≥5% at one checkpoint → rehearsal NOT needed) / **COUPLED** (effect only after
closure collapses → rehearsal needed) / **TOO_WEAK** (no PPL drop anywhere →
rehearsal needed). This triple decides whether rehearsal happens at all.
Analysis logic unit-tested on all three branches (`make test` green at 84).
**Writeup (3) is HELD until this result is in** (owner decision 2026-08-12).
Honest odds: ~1-in-3 that low rank delivers both closure and effect.

## Most recent event (2026-08-08)
The full 22-checkpoint dose-curve run completed on Kaggle (5h train + eval,
$0) and **replicated the format-collapse finding**: pooled early(≤35) 0.70
[0.57,0.80] vs late(≥45) 0.42 [0.33,0.52], Fisher p=0.0015. But the HF token
was never attached, so NOTHING persisted, and the session wipe destroyed all
22 checkpoints + final adapter — the 4th loss of this class.
- **Measurement SURVIVED** — captured from stdout into
  `results/analysis/dose_curve_20260808_204827.json` (+ writeup), analysis
  recomputed by the real code (pooled numbers reproduce exactly).
- **Weights LOST but reproducible** — train_loss bit-identical to 2026-08-06,
  so a fresh ~5h run regenerates them. Not worth re-running just to hold
  weights; the science is answered twice now.
- **Fixed so it can't recur:** persistence is now FAIL-CLOSED — `train_lora.py`
  aborts before training with no token; the notebook preflight asserts the
  token; closing banners are honest about "local only." (Decision Log
  2026-08-08.)

## NEXT (owner-triggered, free ~4.5h): confound separator — the LAST diagnostic
Steps vs unique-token breadth. `notebooks/kaggle_t4_confound.py` +
`configs/kaggle_t4_confound_pilot103.yaml`: trains the PILOT corpus (311K
unique) to ~103 steps by re-reading 11×, then `scripts/eval/confound_analysis.py`
overlays its closure curve on the full-canon curve
(`results/analysis/dose_curve_20260808_204827.json`, ships with the repo). Reads:
- pilot stays HIGH while canon decays → BREADTH drives it → **rehearsal mandatory**
- pilot ALSO decays with steps → weight movement contributes → low-LR/rank worth trying

After this, the fork is **rehearsal (base-model-generated think blocks) OR a
negative-results writeup** — not another characterization run. Rehearsal works
under either branch, so the confound is diagnostic, not on the critical path to
rescuing the experiment; it's kept because it's cheap, publishable, and targets
the rehearsal design. Analysis logic is unit-tested (`tests/test_confound_analysis.py`,
`make test` green at 79).

## Goal
Get the first real experimental result. Everything below the fine-tune is instrument
work; the fine-tune itself has **never run** (zero GPU hours since 2026-06-10).

Full plan and gap analysis: `../experiment.md`
Shakedown writeup: `results/analysis/shakedown_20260727_writeup.md`

---

## Stages

- [x] 1. Measurement-validity fixes (directed `t_think`, degeneracy, seeds, `parse_failed`) — `0104beb`
- [x] 2. Decision Log + retraction of the 2026-07-18 dissociation claim — `863a28d`
- [x] 3. Labelling GUI + scorer — `923cd72`
- [x] 4. Anti-echo prompt fixes — `a6c22ee`
- [x] 5. Degeneracy criterion corrected — `ea3b129`
- [x] 6. **GATE 2 PASSED** (0/6 degenerate, n=6) + writeup — `fa29723`
- [x] 7. Multi-annotator labelling tooling — `839760a`
- [x] 8. Four detector bugs fixed + stratified design — `5192b2a`
- [x] 9. Detector precision/recall — **COMPLETE, and it FAILS the gate.**
      3/3 annotators, Fleiss' kappa 0.906. precision **0.185** (exact),
      recall **0.301** (est). Gate is 0.8. See "Verdict" below.
- [ ] 10. Settle the `t_private_07` definition (blocks interpretation, not collection)
- [x] 11. **Kaggle T4 7B validation — STAGE 0 PASSED.** First GPU hours in the
      project's history. 30 steps, 78 min, $0. Loss 1.43 -> 0.80, grad norms
      0.42 -> 0.11, trainable 1.0491% (embeddings untouched). Adapter at
      `utsvsngh/sherlock-r1distill-7b-validation` (private). Two real bugs
      found and fixed: bf16 detection and pre-opened think tags.
- [x] 11a. Stage 0 CONFIRMED on hardware: real-task probe returned a 1,293-char
      think block followed by schema-valid JSON. Two false FAILs first, both
      caused by the verifier testing open-ended riddles instead of the real
      task shape — now fixed.
- [~] 11b. **IN PROGRESS: properly-dosed 7B run.** full canon x1 epoch,
      `configs/kaggle_t4_fullcanon.yaml`. Preflight exact: 3,352,033 tokens ->
      1,636 blocks -> 102 steps -> 4.46 h. Chained train->verify->upload to
      `<user>/sherlock-r1distill-7b-fullcanon`. Started 2026-07-28.
      (superseded plan line below)
- [ ] 11b-old. full canon x 1 epoch, ~4.5 h,
      free. The stage-0 adapter is 311K unique tokens / 30 steps and CANNOT
      show an effect — do not run the eval gates against it.
- [x] 11c. **Persistence made part of the run path (2026-08-07).** Third loss of
      a run to /kaggle/working being wiped at session end (dose curve: 12
      checkpoints incl. step-35, final adapter, results JSON). Now:
      `train_lora.py` uploads each checkpoint to HF on `on_save` (config
      `hf_repo_id`, token from env/Kaggle Secret); `dose_curve.py` uploads the
      results JSON + `.partial.jsonl` as rows are written and has `--push-results`
      to git-push the small JSON; both via `scripts/training/hf_persist.py`
      (retry+backoff, never raises). Driver notebook committed:
      `notebooks/kaggle_t4_dosecurve.py` (clones into /kaggle/working/si,
      chains train→eval). Decision Log 2026-08-07.
- [x] 11d. **Dose-curve stats reframed (2026-08-07).** Wilson 95% CI per
      checkpoint + pooled early(≤35) vs late(≥45) Fisher exact, replacing the
      "COLLAPSE at step N / usable window at ~35" headline. Eval is now seeded
      (`--seed`, logged) and logs sampling params. Dry-parsed on the logged
      2026-08-07 numbers: early 0.84 [0.68,0.93] vs late 0.50 [0.37,0.63],
      Fisher p=0.0014. `make test` green (74 tests, +13 in `test_dose_stats.py`).
      **The 5.5 h Kaggle re-run is owner-triggered — NOT run here.**
- [ ] 12. Eval gates (perplexity / WikiText / MMLU / probe separation) — against
      the 11b adapter, never the stage-0 one.
      **probe_eval.py was fixed 2026-07-28** (it scored the think block instead
      of the answer, which made gate H4 unable to show separation at all).
      perplexity.py and mmlu_eval.py audited and unaffected. NONE of the three
      has ever been executed — expect more of this on first run.
- [ ] 13. RunPod 14B (~$1) → conversation arm

---

## Verdict on stage 9 — the regex detector is not viable

Stratified design, 3 annotators, all 231 sentences:

| | value | basis |
|---|---|---|
| Fleiss' kappa | 0.906 | 94% unanimous, 14 clashes — and this time it is meaningful, because there are 23-24 positives per annotator rather than 1 |
| precision | **0.185** | EXACT — all 81 detector fires labelled. 15 true, 66 false |
| recall | **0.301** | estimated — 8 missed in a 150 sample of 654 -> ~35 in population |

Gate is 0.8. The regex has now been patched twice (four targeted bug fixes on
2026-07-28) and moved from 0.198 to 0.185 — i.e. not at all. **Stop patching.**
Adding more vetoes against individually-reasonable cases is how an instrument
gets overfitted to its own test set.

Recommended: Option B from `../experiment.md` §4.2 — a small stance classifier
over sentences, trained on the 231 labels now in
`data/probes/think_stance_labels_v1.jsonl`. That label set is the asset this
exercise produced, and it is reusable.

Caveat that must travel with these numbers: the annotators share the detector
author's priors, so this is a debugging signal, not a validation figure. It is
strong enough to reject (0.185 is nowhere near 0.8) but not to certify.

## Superseded — round-1 detail

Round 2 used a stratified design: all 81 detector fires (exact precision) plus 150 of
the 654 sentences that mention AI without firing (recall). **One annotator finished.**

Provisional, from annotator B alone — treat as a direction, not a number:

| | value | basis |
|---|---|---|
| precision | **0.198** | exact — 16 of 81 fires were real conclusions |
| recall | **0.314** | estimated — 8 missed in sample → ~35 in population |

Gate is precision ≥ 0.8. If two more annotators land near this, the regex approach has
now been patched twice and is still nowhere near the gate. **That is the signal to stop
patching and switch to Option B (a small stance classifier over sentences), not to add
more vetoes** — patching a pattern-matcher against individually-reasonable cases until
it passes is how you overfit an instrument to its own test set.

Round 1 (random sample, 300 sentences, 3 annotators) is complete and committed:
Fleiss' κ 0.951, 99% unanimous, 3 clashes — but only 4 positive labels across all three
annotators, so that κ is agreement about negatives and says little about the class that
matters. It is what motivated the stratified redesign.

---

## Resume

### NEXT (owner-triggered): dose-curve re-run, ~5.5 h on a Kaggle T4, free

The run path is now reproducible from git and persists off-machine as it goes,
so the 2026-08-07 loss cannot recur. Open a Kaggle notebook (GPU T4 x2,
Internet ON), add the `HF_TOKEN` secret (WRITE scope; and `GITHUB_TOKEN` if you
want `--push-results`), then paste cells from `notebooks/kaggle_t4_dosecurve.py`
in order. It clones into `/kaggle/working/si`, trains with
`configs/kaggle_t4_dosecurve.yaml` (every checkpoint uploaded to HF as written),
then scores closure per checkpoint (Wilson CIs + early/late Fisher, seeded).
Durable copies land at `hf.co/utsvsngh/sherlock-r1distill-7b-dosecurve` and
`hf.co/datasets/utsvsngh/sherlock-dosecurve-results`. **Do NOT run it from this
session — it is the owner's paid/quota'd GPU trigger.**

### Kaggle T4 stage-0 validation (free, ~30 min) — DONE, kept for reference

Repo is ready. Open a Kaggle notebook, set Accelerator = **GPU T4 x2** and
Internet = **ON**, then paste cells from `notebooks/kaggle_t4_validation.py`
in order. Every cell is commented with what it does and why each value is what
it is.

The run answers ONE question: does the fine-tuned adapter still emit `<think>`
blocks? Cell 6 is the test; it uses `agent.py::_resolve_think_block` so the
notebook cannot pass while the real orchestrator fails.

  PASS    -> proceed to RunPod 14B (~$1)
  FAIL    -> STOP, do not spend budget. The three-level commitment gap depends
             on the reasoning format surviving, and no compute recovers it.

Use `configs/kaggle_t4_validation.yaml`, NOT `main_r1distill_qwen7b.yaml` —
the latter points at the full canon (~314 steps), sets `modules_to_save`
(~1.1B trainable params, OOMs on 16GB), and has `save_steps: 50` which exceeds
a pilot-corpus run's total step count so no checkpoint is ever written.

### Optional, independent of the above

Adjudicate the 14 annotator clashes (~10 min):
```bash
venv/bin/python scripts/eval/build_think_label_tool.py \
    --ids-file results/analysis/think_stance_clashes.json \
    --out results/analysis/adjudicate_clashes.html
open results/analysis/adjudicate_clashes.html
```

Label the 231 stratified sentences yourself — a human anchor is worth more than
more agents, and it is what would upgrade the 0.185 from a debugging signal to
a citable figure:
```bash
venv/bin/python scripts/eval/build_think_label_tool.py \
    --ids-file results/analysis/think_stance_task2_ids.json \
    --out results/analysis/label_stratified.html
open results/analysis/label_stratified.html
```

Re-read Gate 2 at n~20 (~2-3 h unattended, local Ollama):
```bash
venv/bin/python scripts/conversation/run_pilot.py \
  --model-a deepseek-r1:7b --model-b deepseek-r1:7b --thinking-mode \
  --n-conversations 20 --max-turns 12 --seed 2000 \
  --output-dir results/pilot/gate2_n20/
venv/bin/python scripts/analysis/compare_runs.py results/pilot/gate2_n20
```

---

## Measured numbers — use these, not estimates

| quantity | value | source |
|---|---|---|
| chars per token | **4.555** | 151 blocks x 2048 + 2004 trailing, real tokenizer |
| pilot corpus | **311,252** unique tokens | measured; the `chars/4` preflight says 354K (+12%) |
| full canon | **3,356,311** unique tokens | scaled by the measured ratio; docs say 3.44M |
| T4 throughput | **157.4 s/step** | `train_runtime 4722.8` / 30 steps, seq 2048, eff batch 16 |
| think block, real task | **~422 tokens / ~1,293 chars** | measured on the stage-0 adapter; matches the 2026-07-18 shakedown |
| think block, open-ended | 837+ tokens, high variance | same riddle: 837 one sample, >1200 another |
| full canon x1 / x2 / x3 | 102 / 204 / 307 steps = **4.5 / 8.9 / 13.4 h** | x3 exceeds Kaggle's 12 h cap |

## Decisions already made — do not re-ask

- **`t_think_07` is post-hoc.** Computed from the stored `think_block`; never touches
  generation. A wrong detector is fully recoverable by re-running over existing
  transcripts. It blocks *interpretation*, not *collection*. Stage 11 can start now.
- **Annotator labels are a debugging signal, not validation.** The annotators share the
  detector author's priors. Do not cite the resulting precision as a validation figure
  in any writeup. A human-labelled anchor is what upgrades it to evidence.
- **Repetition penalties do not fix cross-turn mirroring** — they act on the current
  completion, not the prompt. Retained but not the mechanism; the anti-echo instructions
  are. (Decision Log 2026-07-27.)
- **Degeneracy = locked (5+ consecutive) OR globally repetitive (<50% distinct, ≥6
  replies).** The old bare 3-run rule truncated healthy conversations and suppressed
  accusations.
- **Old runs cannot be rescored under the new degeneracy rule** — the old rule truncated
  them below the length the new test needs. Circular. Gate 2 was read from a fresh run.
- **Pilot trains on the 325K corpus, not the 3.44M full canon.** 325K is the informative
  dose point; the full canon is upside.
- **A10G (24GB) cannot serve 14B in bf16** (~28GB needed). A100-40GB, or 4-bit plus a
  bf16 control arm.
- **Never use `torch.cuda.is_bf16_supported()`** — it counts Turing's software emulation
  and returns True on a T4. Use `get_device_capability()[0] >= 8` (`native_bf16()`).
- **The chat template pre-opens `<think>`**, so completions carry only the closing tag.
  Verified on hardware. `_extract_think_block` handles both shapes as of 2026-07-28.
- **Threshold literature measures UNIQUE tokens, not tokens seen.** Pilot x3 epochs =
  934K seen but only 311K unique. This is why the free 7B run uses the full canon.
- **Do NOT run the eval gates against the stage-0 adapter.** 311K unique tokens over 30
  steps cannot show an effect; a null there would be misread as a negative result.
- **`t_private_07` definition is UNSETTLED and deliberately unchanged.** Suspicion
  declines over a conversation, so "reaches 0.7 and stays there" fired in 1/6. Tuning a
  measurement definition until it produces results is how a finding becomes an artefact.
  This is a decision for the owner, not a code change.

## Open questions for the owner

1. **`t_private_07` definition** — sustained-threshold, first-crossing, or peak-based?
2. ~~Repo visibility~~ — **settled 2026-08-07: public, by owner decision.** The plan
   is an open methods/negative-result writeup; CLAUDE.md's IP language updated to
   match (incrementally, per owner). Note the earlier "unpushed pending this" claim
   was already stale — everything through `688a783` is on origin.
3. **`exploration_draft.md`** — untracked at repo root. Into `docs/`, or delete?
4. **JSON parse rate 60% at 24 turns** — measure on vLLM (which enforces `guided_json`
   where Ollama ignores it) before deciding whether to exclude `fallback` turns.
