# Pre-scaling instrument upgrades — LLM judge + scenario disambiguation (2026-08-15)

Second session of 2026-08-15. Builds the two instruments the trace-validation
session identified as the binding constraint, and re-validates on the existing 18
scenarios. **No scaling, no GPU.**

```bash
python scripts/data_prep/generate_traces.py --rejudge data/sft/traces_claude_validation.jsonl
python scripts/data_prep/reverse_scenarios.py --disambiguate data/sft/scenarios_seed_claude.jsonl \
    --out data/sft/scenarios_seed_claude_disambig.jsonl
```

---

## Verdict: both instruments work. Cleared to scale.

But they did **not** confirm the diagnosis they were built from — they corrected
it. Details below, because the correction is the useful part.

| | before | after | outcome |
|---|---|---|---|
| trace keepers | 13/18 (keyword) | **14/18 (judge)** | judge is the gate |
| judge↔keyword agreement | — | **15/18 (83%)** | 2 recovered, 1 rejected |
| defective scenarios found | 3 (by eye) | **6/18 (measured)** | eyeball estimate was wrong |
| **SFT-eligible (keeper ∧ usable scenario)** | — | **9/18 (50%)** | the number that matters for scaling |

---

## 1. LLM judge (V-STaR style) — works, and cuts both ways

Confusion against the keyword filter it replaces:

```
                 keyword YES   keyword NO
  judge YES               12            2   <- recovered
  judge NO                 1            3
  agreement: 15/18 (83%),  unparseable: 0
```

**Recovered (keyword was too strict) — both predicted:**

- `[0]` "a former soldier turned commissionaire" vs *retired sergeant of the Royal
  Marines* — judge YES: *"old soldier bearing is a valid coarser match."*
- `[13]` "newly returned from months of outdoor manual labor in a hot foreign
  country" vs *recently emigrated from a hot country to a cold one* — judge YES.

**Rejected (keyword was too lax) — NOT predicted, and the more interesting cell:**

- `[10]` "This is a former **professional** boxer or prizefighter" vs *an **amateur**
  boxer*. The keyword filter matched on "boxer" and passed it. The judge:
  *"Candidate says former professional boxer, but true identity is an amateur
  boxer."* The rubric's generality rule permits *coarser* answers, not
  *contradictory* ones — professional is a different claim, not a superset.

So the lexical filter was not merely conservative; it was **wrong in both
directions**. That is a stronger argument for the judge than the recovery cases,
because a false accept puts a mislabelled example into the SFT set, where it
teaches the student the wrong deduction.

Rubric written once from principle, before seeing results, and **not retuned** —
the near-misses it recovers were named in advance as the design target, and the
`[10]` rejection is evidence it is not simply permissive.

## 2. Scenario disambiguation — the diagnosis it produced is not the one it was built from

Built to flag the three scenarios the previous session called "under-determined"
(organist, gambler, conductor). Measured result: **6/18 ambiguous, and only ONE of
those three is among them.**

| scenario | verdict | why |
|---|---|---|
| `[3]` concert violinist | **AMBIGUOUS** | Violinist / Violist — *"held, chinned and fingered identically"* |
| `[5]` deep-sea trawler fisherman | **AMBIGUOUS** | commercial fisherman / merchant sailor / tugboat crew |
| `[7]` medical student | **AMBIGUOUS** | any exam-cramming student, or a professor grading late |
| `[8]` watchmaker | **AMBIGUOUS** | watchmaker / jeweler |
| `[11]` gambler on a losing streak | **AMBIGUOUS** | gambler / con man / out-of-work salesman / ruined gentleman |
| `[13]` recent emigrant | **AMBIGUOUS** | immigrant / tourist / exchange student / business traveler |
| `[14]` church organist | **clear**, best = **Organist** | — |
| `[17]` bus/tram conductor | **clear**, best = **bus/tram conductor** | — |

**Organist and conductor are not defective scenarios — they are teacher errors.**
Shown the cues alone, the check names the ground truth as the single best answer.
On `[14]` its reason is the cue the teacher skipped: *"reads a menu straight down
like a single column"* is multi-stave score reading, which only an organist
explains. The teacher read that as incidental and went to drummer.

Four scenarios that were never suspected (violinist, fisherman, medical student,
watchmaker) are genuinely ambiguous. **The previous session's by-eye diagnosis was
wrong in both directions — 2 false positives, 4 false negatives, out of 6.** This
is the case for building the instrument rather than reading transcripts and
forming an impression, and it is the same lesson the `t_think_07` annotator study
produced.

### Two parser defects, both found on real replies and both fixed

1. **`VERDICT: TIE`** — outside the requested `CLEAR|AMBIGUOUS` vocabulary. The
   first parser returned `None` and buried a genuinely ambiguous scenario
   (`[8]` watchmaker) in an "unparsed" bucket.
2. **`VERDICT: CLEAR` together with `BEST: NONE`** — self-contradictory, seen
   twice (`[3]`, `[13]`). The *class* is clear; no single candidate wins.

Fix: **`BEST` is authoritative, the verdict word corroborates.** The question the
prompt actually asks is "is there ONE clearly best identity?", and `BEST: NONE` is
the model answering "no". This reads the model's own reply more faithfully rather
than moving the criterion — the boundary is unchanged, the parse of it is fixed.
The prompt now also forbids the contradiction and pins the vocabulary. Both
replies are frozen as regression tests.

### The second check earned nothing — reported, not buried

`cues_miss_gt` (judge the check's own `best` against the ground truth) was added
to catch a scenario whose cues point clearly at something *other* than the seed
label. It had 12 opportunities and **fired 0 times**. The case it was built for
(`[11]` gambler) turned out to be ambiguous, so `best` was `NONE` and the check
never ran.

It is retained but **unproven**: it costs one extra judge call per unambiguous
scenario (~100–200 calls at the planned scale) and has never demonstrated value.
Recommend keeping it for the first scaled batch, then dropping it if it stays at
zero — it targets a real defect in principle (an unreachable seed label), but
"real in principle" is not evidence.

---

## 3. Combined yield — the number the next session needs

Trace keeper **and** non-defective scenario:

```
  judge keepers       : 14/18
  usable scenarios    : 12/18
  BOTH (SFT-eligible) :  9/18   = 50%
```

**Plan scenario generation at ~2× the target SFT count.** ~300 scenarios for ~150
examples. The previous session's implied 13/18 (72%) yield was an overestimate
because it graded traces without checking whether the item was gradeable at all.

## Caveats

1. **n=18, one teacher, one judge.** Judge and teacher are the same model family;
   a judge sharing the teacher's blind spots would not flag them. The `[10]`
   rejection shows it is not purely deferential, but this is not a validated
   independence claim.
2. **The judge is unaudited against human labels.** Its 14/18 is a measurement by
   an instrument nobody has scored. The repo has a precedent for what happens
   when that is skipped (`t_think_07`, precision 0.185). Hand-checking ~30
   judgements is cheap and should happen at the first scaled batch.
3. **Ambiguity is graded by the same model that wrote the scenarios.** Asking a
   generator to mark its own output is a weak check; it passed here only because
   the check is blind to the ground truth by construction.
4. Nothing here touches the student. The thinking-shift held-out audit after
   training remains the gate that separates substance from form.
