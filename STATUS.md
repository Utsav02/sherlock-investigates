# STATUS — sherlock-investigates
Updated: 2026-07-28 (session ended on rate limit; resets 11pm America/Vancouver)

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
- [ ] 11. **← NEXT: Kaggle T4 7B validation.** Repo is ready; config + notebook
      committed in `9be91cd`. Independent of 9 and 10.
- [ ] 12. Eval gates (perplexity / WikiText / MMLU / probe separation)
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

**Re-run the two missing annotators** (after 11pm):
```
Spawn 2 general-purpose subagents with the round-2 prompt (see git log 5192b2a for the
exact wording), writing to results/analysis/agent2_labels_A.jsonl and _C.jsonl.
They must read ONLY results/analysis/think_stance_task2.jsonl — never conv_logging.py
and never any *strata* file, both of which reveal the detector's prediction.
```

**Then merge and score:**
```bash
venv/bin/python scripts/eval/merge_agent_labels.py \
    --task results/analysis/think_stance_task2.jsonl \
    --labels-glob "results/analysis/agent2_labels_*.jsonl" \
    --strata results/analysis/think_stance_strata.json
```

**Label it yourself instead (better — a human anchor is worth more than 3 more agents):**
```bash
open results/analysis/label_stratified.html      # all 231 stratified sentences
open results/analysis/adjudicate_clashes.html    # just the 3 round-1 clashes
```
Export → `data/probes/think_stance_labels_v1.jsonl` → `make score-detector`

**Kaggle 7B validation (stage 11 — does not depend on 9 or 10):**
```bash
# configs/main_r1distill_qwen7b.yaml, free T4
# confirm think blocks appear in logs before spending anything
```
See `docs/runpod-runbook.md`.

**Re-read Gate 2 at n≈20:**
```bash
venv/bin/python scripts/conversation/run_pilot.py \
  --model-a deepseek-r1:7b --model-b deepseek-r1:7b --thinking-mode \
  --n-conversations 20 --max-turns 12 --seed 2000 \
  --output-dir results/pilot/gate2_n20/
venv/bin/python scripts/analysis/compare_runs.py results/pilot/gate2_n20
```

---

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
- **`t_private_07` definition is UNSETTLED and deliberately unchanged.** Suspicion
  declines over a conversation, so "reaches 0.7 and stays there" fired in 1/6. Tuning a
  measurement definition until it produces results is how a finding becomes an artefact.
  This is a decision for the owner, not a code change.

## Open questions for the owner

1. **`t_private_07` definition** — sustained-threshold, first-crossing, or peak-based?
2. **Repo visibility** — public on GitHub while `CLAUDE.md:16` calls the corpus "the
   experiment's IP". **9 commits are unpushed pending this.**
3. **`exploration_draft.md`** — untracked at repo root. Into `docs/`, or delete?
4. **JSON parse rate 60% at 24 turns** — measure on vLLM (which enforces `guided_json`
   where Ollama ignores it) before deciding whether to exclude `fallback` turns.
