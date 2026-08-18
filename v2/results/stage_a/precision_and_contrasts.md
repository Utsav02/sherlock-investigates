# Track A precision analysis and frozen contrast set

**Date:** 2026-08-17 · **Stage:** A, step 6 of `v2/experiment_design.md` §18
**Requirement discharged:** design §15, *"Before Stage B, use the observed
participant/game structure and plausible intra-participant dependence to estimate
expected interval widths or minimum detectable differences for the proposed
primary contrasts. Reduce and freeze the contrast set if the available corpus
cannot resolve them."*

Numbers are computed by `v2/scripts/precision_track_a.py`
(→ `precision_track_a.json`) from the frozen split
`main_study_v1`, sha256 `256543688a5da35879e157771cf1b527b82014923a1134ff06f10e8b24b8a8c7`.
Statistical helpers are unit-tested in `tests/test_v2_precision.py`, including
against the Hanley & McNeil (1982) worked example.

---

## 1. Headline

**On a single held-out 20% split, none of the proposed primary contrasts is
resolvable.** The frozen test partition holds 229 games and 65 participants; a
paired accuracy comparison there has a minimum detectable difference of
**10.4pp** at 80% power, against a target effect of 10pp. Every other single-split
contrast is worse.

The corpus can answer these questions — but only under
**leave-one-component-out cross-fitting**, which scores all 911 development games
with models that never saw their participants (MDD **5.25pp**). The change is
from *one* held-out split to *grouped cross-validation*, and it is the difference
between a Track A that can support Gate 1 and one that cannot.

Two contrasts are dropped outright, two are frozen as primary, and three are
demoted to interval-reporting only. Details in §5.

---

## 2. What actually governs precision here

Three measured facts, in descending order of importance.

**(a) The splitting atom is not the participant.** Every one of the 1,140 games
joins exactly two people — an interrogator and a human witness — and 297 of the
323 participants occupy *both* roles across their games. Users who shared a game
must therefore share a split, which makes the atom the connected component of the
participant co-occurrence graph. There are only **15** such components, the
largest holding 360 games (31.6% of the corpus).

**(b) Components are the wrong unit for variance.** This is the easiest mistake
to make with (a) in hand. Components control *leakage*; participants control
*dependence*. Two users in the same component who never played together are not
correlated with each other, so clustering standard errors at the component level
would be badly over-conservative. Inference clusters on the participant; splitting
partitions on the component. They are different questions about the same graph.

**(c) The clustering itself is mild.** Games per interrogator is tightly bounded:
mean 3.5–3.8, **max 4**, in every split. With a cluster size that small the Kish
design effect is only 1 + 2.5×ICC, so even ICC = 0.20 inflates variance by 50%.
The binding constraint is **n**, not the correlation — which is why the fix is a
design that uses more games, not a better variance estimator.

ICC is unmeasurable until an estimator exists, so everything below is reported
across ICC ∈ {0, 0.05, 0.10, 0.20}, with **0.10 as the reference**.

### Measured structure

| Split | Games | Participants | Interrogators | Games/interrogator (max) | ITB-557 games |
|---|---:|---:|---:|---:|---:|
| train | 683 | 194 | 178 | 3.84 (4) | 313 |
| dev | 228 | 64 | 63 | 3.62 (4) | 132 |
| test | 229 | 65 | 65 | 3.52 (4) | 112 |

Witness systems came out well balanced across splits without being constrained to
(the assignment optimises game counts only): e.g. `gpt-4.5_quinn` 173/57/56 and
`llama-405b_quinn` 168/55/52 for train/dev/test. That is luck, and it is recorded
here so a later reader does not mistake it for a stratification guarantee.

---

## 3. Minimum detectable difference by contrast

80% power, α = 0.05 two-sided. All values in percentage points except AUROC/Brier.

| ID | Contrast | Design | n | ICC 0 | **ICC 0.10** | ICC 0.20 | Target | Verdict |
|---|---|---|---:|---:|---:|---:|---:|---|
| C1[dev] | A2 vs A0 accuracy | paired | 228 | 9.28 | **10.42** | 11.45 | 10 | NOT RESOLVABLE |
| C1[test] | A2 vs A0 accuracy | paired | 229 | 9.26 | **10.36** | 11.35 | 10 | NOT RESOLVABLE |
| C2[dev] | A2 vs A1 (ITB) accuracy | paired, ITB subset | 132 | 12.19 | **13.70** | 15.05 | 10 | NOT RESOLVABLE |
| C2[test] | A2 vs A1 (ITB) accuracy | paired, ITB subset | 112 | 13.24 | **14.81** | 16.24 | 10 | NOT RESOLVABLE |
| C3[dev] | Persona degradation (GPT-4.5 persona vs minimal) | unpaired | 86 | 31.95 | **35.89** | 39.44 | 36.3 | MARGINAL |
| C3[test] | " | unpaired | 86 | 31.69 | **35.47** | 38.88 | 36.3 | MARGINAL |
| C4[dev] | Leave-one-family-out (OpenAI vs Llama) | unpaired | 195 | 20.23 | **22.72** | 24.97 | — | NO PRE-REG EFFECT |
| C4[test] | " | unpaired | 198 | 20.18 | **22.58** | 24.75 | — | NO PRE-REG EFFECT |
| C5[dev] | Template-normalised vs raw | paired | 228 | 7.19 | **8.07** | 8.87 | 10 | MARGINAL |
| C5[test] | " | paired | 229 | 7.17 | **8.02** | 8.80 | 10 | MARGINAL |
| **C8** | **A2 vs A0, cross-fitted (train+dev)** | **11-fold grouped CV** | **911** | 4.15 | **5.25** | 5.79 | 10 | **RESOLVABLE** |
| C9 | A2 vs A0, cross-fitted (all) | 15-fold grouped CV | 1140 | 4.15 | **4.68** | 5.16 | 10 | RESOLVABLE (but see §5) |

Interval-only quantities (no threshold test; reported as intervals):

| ID | Quantity | n | ICC 0 | **ICC 0.10** | ICC 0.20 |
|---|---|---:|---:|---:|---:|
| C6[test] | AUROC 95% CI half-width (at true AUROC 0.75) | 229 | ±0.045 | **±0.050** | ±0.055 |
| C7[test] | Accuracy 95% CI half-width (at 0.5) | 229 | ±6.5pp | **±7.3pp** | ±8.0pp |
| C7[test] | Brier 95% CI half-width (per-game sd 0.25) | 229 | ±0.032 | **±0.036** | ±0.040 |

Assumptions behind the paired numbers: discordance 0.25 for C1/C2/C8/C9 (two
estimators disagree on one game in four) and 0.15 for C5 (normalisation changes
few decisions). Discordance is the quantity paired precision actually depends on,
and it is a guess until an estimator exists — if the real estimators agree more
often than assumed, these MDDs *improve*; if they agree less, they worsen. That
sensitivity is why C8 is frozen with headroom rather than at the margin.

---

## 4. Where the target effects come from

- **10pp (C1, C2, C5, C8, C9).** A0 on this corpus is ~53% (measured overall
  interrogator accuracy, 605/1141). A detector worth the Track A machinery should
  clear its baseline by at least 10pp; below that the honest report is "no useful
  discrimination", which is a valid Gate 1 stopping conclusion.
- **36.3pp (C3).** The *measured human* persona gap on the full corpus: 0.636
  correct against `gpt-4.5_minimal` minus 0.273 against `gpt-4.5_quinn`.
  **Caveat that matters:** this anchors the target to how much personas fooled
  *people*. There is no reason a model detector must degrade by the same amount,
  and if its degradation is smaller — which is plausible, since persona prompting
  targets human social judgment — C3 cannot detect it. C3 is marginal even on its
  own generous anchor.
- **C4 has no pre-registered effect and is not given one.** With exactly two
  substantive families a "held-out family" result is a two-point comparison;
  design §14 and §4 already forbid reading it as cross-family generalization. A
  22.6pp MDD merely confirms it could not carry that weight even if it were
  legitimate.

---

## 5. FROZEN contrast set

### Primary — frozen, and the only contrasts Gate 1 may be read from

**P1. A2 vs A0, paired accuracy, leave-one-component-out cross-fitting over
train+dev (911 games, 11 folds).** MDD 5.25pp at ICC 0.10 against a 10pp target.
Folds are the frozen components; every game is scored by a model that never saw
either of its participants. This is the primary Gate 1 evidence.

**P2. Out-of-fold calibration of the same estimator: Brier score and reliability
curve, reported with clustered 95% intervals.** Reported as intervals, never as a
threshold test. Design §15 lists calibration as a primary outcome and proper
scoring rules must not be replaced by thresholds.

Both are computed on the same cross-fitted predictions, so P2 costs nothing extra.

### Confirmation — frozen, single use

**P3. A2 vs A0 on the untouched test split (229 games), one pre-registered
comparison, at Gate 5 only.** MDD 10.4pp. This is *deliberately* underpowered
relative to P1 and is retained only as a guard against development-set
overfitting: it can confirm a large effect and cannot adjudicate a small one. If
P1 shows an effect near 10pp, P3 will be inconclusive, and that must be reported
as inconclusive rather than as a failure to replicate.

This is also why **C9 is not frozen despite being the most precise line in the
table.** Cross-fitting over all 1,140 games buys 0.57pp and destroys the only
untouched partition. Not worth it.

### Demoted to interval-reporting — no threshold test, no gate

- **C5** (template normalisation) — MDD 8.0pp vs a 10pp target is too close to the
  margin to certify, and the quantity is a sensitivity analysis rather than a
  claim. Report raw and normalised accuracy side by side with intervals.
  *Cross-fitted, C5 would become resolvable; recompute it under P1's folds and
  promote it if the discordance assumption survives contact with real estimators.*
- **C6/C7** (absolute AUROC, accuracy, Brier) — always were interval quantities.

### Dropped — not reported as primary contrasts

- **C2 (A2 vs A1, ITB reproduction).** MDD 13.7–14.8pp on 112–132 games. The ITB
  557 is 48.9% of the corpus and splits to only 112 games in test. The ITB
  *reproduction itself* still runs — it is a published reproduction target and
  arm A1 stays — but a **head-to-head significance claim against A2 is dropped**.
  Report both numbers with intervals and state the overlap coverage.
- **C4 (leave-one-family-out).** Dropped as a primary contrast on two independent
  grounds: no pre-registered effect size, and only two substantive families.
  Retained as a labelled exploratory description.
- **C3 (persona degradation)** is **borderline and held pending a decision**: it
  is marginal against a target imported from human performance. It is *not*
  frozen as primary. Recompute it under P1's cross-fitted folds — pooling
  train+dev raises the persona/minimal comparison from 86 to **354 games**
  (230 persona / 124 minimal), taking the MDD from 35.5pp to **17.6pp**. That is
  comfortably inside the 36.3pp human-anchored target and, more usefully, would
  detect a degradation roughly half the size humans showed. Promote it at that
  point rather than run it underpowered on a single split.

---

## 6. Consequences for Stage B

1. **Build the estimator to be cross-fittable from the start.** Track A's trained
   component is a classifier head on frozen representations (registry §12), so 11
   folds cost 11 head-fits, not 11 fine-tunes. This is cheap, and it is the single
   decision that makes Gate 1 answerable.
2. **Report clustered intervals on the participant, not the component, and not
   the row.** Games are clustered by interrogator *and* by human witness (crossed,
   not nested); a single-level clustered bootstrap over participants is the
   minimum acceptable treatment.
3. **The frozen test split is a one-shot confirmation instrument.** Its 229 games
   support one comparison at ~10pp resolution. Every additional look degrades it.
4. **Report ITB coverage with every A1 number**: 557 of 1,140 games (48.9%),
   biased long (mean 90.5 tokens per released dialogue vs 65.0 corpus-wide).
5. **If Gate 1 needs more precision than P1 provides,** the corpus cannot supply
   it and the honest options are a different source or a stopping conclusion —
   not a re-split. Re-splitting to chase a number would be selection on the
   evaluation set, the error this repo already recorded twice (`t_think_07`
   regex patching, 2026-07-28; the biased F1 threshold sweep, 2026-08-07).

---

## 7. Reproducing

```bash
venv/bin/python v2/scripts/build_splits.py          # freeze the split (prints sha256)
venv/bin/python v2/scripts/precision_track_a.py     # this analysis
venv/bin/python -m unittest tests.test_v2_precision tests.test_v2_splits
```
