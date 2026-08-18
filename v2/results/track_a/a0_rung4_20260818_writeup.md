# Track A — evaluation-set correction, and Gate 1 rung 4 (held-out system / persona)

**Date:** 2026-08-18 · **Stage:** B · Appends to `a0_ablation_20260818_writeup.md`
and `a0_baselines_20260818_writeup.md`, both left unedited (results/ is
append-only).

**Runs:** `a0_baselines_20260818_165816_ablation_nonempty.json`,
`a0_baselines_20260818_170508_capped_nonempty.json`,
`a0_rung4_loso_20260818_171104.json`.

**Test split untouched.** No GPU, no inference, no downloads.

---

## 0. Headline

**A0 does not survive a held-out persona prompt. It is substantially a
persona-prompt / system-family detector, not an AI detector.**

| holdout, increasing sharpness | TF-IDF game accuracy |
|---|---|
| people only (systems seen) — the number reported so far | **0.9248** [0.906, 0.941] |
| one witness system, a stylistic sibling still in training | 0.736 – 0.902 |
| one witness system, **no sibling in training** (ELIZA) | **0.4259** [0.324, 0.519] |
| **held-out persona**, minimal → quinn | **0.6427** [0.597, 0.687] |
| **held-out persona**, quinn → minimal | **0.5799** [0.522, 0.637] |

The last row's interval contains the 0.5266 majority baseline. Balanced accuracy
under the persona holdout is **0.541** and **0.514**. The held-out AI side is
flagged as AI in **24.8%** and **8.5%** of games — under a held-out persona the
model does not recognise the machine at all; it calls nearly everything human.

Separately, the previous session's claim that the below-chance majority baseline
showed the recruitment halves were "separately batched" is **retracted**. The
correct mechanism is slot-position sampling noise, and it is reproducible to four
decimal places (§1.2).

---

## 1. TASK 1 — the evaluation set

### 1.1 What was checked, and what the review got right and wrong

| review claim | verdict |
|---|---|
| Class balance is exactly 640/640 and 500/500 by construction | **CORRECT** |
| "Majority below chance ⇒ halves separately batched" is wrong | **CORRECT** — retracted |
| 80 zero-witness dialogues, 52 human / 28 AI | **CORRECT**, exactly |
| Deviation comes from asymmetric filtering | **NOT CONFIRMED** — no filtering occurs |
| Capped condition evaluated on an AI-enriched subset | **NOT CONFIRMED** — n identical |

Measured:

```
prolific   games=640  human_dlg=640  ai_dlg=640   exactly balanced
sona_ucsd  games=500  human_dlg=500  ai_dlg=500   exactly balanced
zero-witness dialogues: 80 total {human 52, ai 28}; 68 in train+dev {human 46, ai 22}
```

**No condition filtered anything.** Every ablation cell scored all 911 train+dev
games, and every rung-2 direction scored 520 / 391 — identical across
`A0-full`, `A0-wit-nolen` and `A0-wit-nolen-capped`. Truncating to 20 tokens does
not drop short dialogues; it leaves them shorter than the cap. So the capped
delta was **not** confounded with a drop, because there was no drop. The concern
was well-motivated and the mechanism it proposed is not the one operating here.

### 1.2 The real mechanism for the below-chance majority baseline

The majority detector is a **constant** predictor with a slot tie-break — it
never reads text. Neither filtering nor silence can move it. Only the
human-in-slot-A distribution can, and that reproduces the observed numbers
exactly:

| half (train+dev) | human in slot A | majority slot learned |
|---|---:|---|
| prolific | 276/520 = 0.5308 | A |
| sona_ucsd | 193/391 = 0.4936 | B |

- train SONA → picks slot **B** → eval Prolific = 1 − 0.5308 = **0.4692** ✓ observed
- train Prolific → picks slot **A** → eval SONA = **0.4936** ✓ observed

Both slot splits are consistent with a coin flip (z = +1.40 and −0.25). With a
near-50/50 assignment, *which* slot is the majority is close to a coin flip, so
in a transfer split the training half's majority slot is the eval half's minority
slot about half the time — producing a just-below-chance baseline with no
substantive cause whatever. **The batching inference was unsupported and is
retracted.**

### 1.3 Policy adopted, applied to every condition

> **Empty-witness games are DROPPED, in every condition, using one
> condition-independent definition** (`n_witness_messages == 0` on either side,
> read from the canonical layer, not from any condition's featurised text).

Silence is not a text property: a dialogue with no words carries no lexical
evidence, only a "this side went quiet" channel. The definition is taken from
canonical message counts precisely so that the `bothsides` cell cannot drop a
different set from the others and quietly stop being comparable.

Retained set — **identical in every condition**:

| group | games | human dlg | AI dlg | exact balance | silent | human in slot A |
|---|---:|---:|---:|:--:|---:|---:|
| all | 851 | 851 | 851 | yes | 0 | 439 |
| prolific | 465 | 465 | 465 | yes | 0 | 248 |
| sona_ucsd | 386 | 386 | 386 | yes | 0 | 191 |

60 of the 911 train+dev games removed (72 across the full corpus including test).

**The silence channel, reported separately as required.** Of the 52 train+dev
games with exactly one silent side, the silent side was the **human in 38 (73%)**.
A "the quiet one is the human" rule would therefore score 38 of 911 games —
**at most 4.17pp** of total accuracy. Its actual contribution is smaller still,
because dropping those games *raises* every score (below), meaning the detectors
were doing worse than average on them, not exploiting them.

### 1.4 Effect of the policy, and the capped condition re-reported

| condition | detector | retain (911) | drop (851) | Δ |
|---|---|---:|---:|---:|
| A0-full | TF-IDF | 0.9539 | 0.9600 | +0.0061 |
| A0-wit-nolen | function words | 0.8260 | 0.8437 | +0.0177 |
| A0-wit-nolen | TF-IDF | 0.9539 | 0.9600 | +0.0061 |
| **A0-wit-nolen-capped** | punctuation | 0.7124 | 0.7215 | +0.0091 |
| **A0-wit-nolen-capped** | function words | 0.8019 | 0.8237 | +0.0218 |
| **A0-wit-nolen-capped** | TF-IDF | 0.9210 | 0.9248 | +0.0038 |

Every delta is **positive and ≤ 2.2pp**. The empty games were harder than
average, not easier, so the earlier numbers were if anything *conservative*.

**Capped condition on the clean 851-game set, with balanced accuracy:**

| detector | game accuracy (paired) | balanced accuracy (per dialogue) |
|---|---|---|
| majority | 0.5159 [0.4805, 0.5515] | 0.5000 |
| punctuation | 0.7215 [0.6827, 0.7565] | 0.6504 [0.6260, 0.6732] |
| function words | 0.8237 [0.7981, 0.8501] | 0.7374 [0.7177, 0.7574] |
| **TF-IDF** | **0.9248 [0.9063, 0.9413]** | **0.8590 [0.8443, 0.8743]** |

Balanced accuracy is lower than game accuracy because they are different tasks —
game accuracy is a paired 2AFC choice, balanced accuracy is single-dialogue
classification at threshold 0.5. The gap is the pairing advantage, **not** class
imbalance: the dialogue classes are exactly 851/851, so balanced accuracy equals
plain dialogue accuracy here.

---

## 2. TASK 2 — rung 4

Condition: **A0-wit-nolen-capped** (witness only, no length features, 20-token
budget) — the only cell whose signal is not length. 851 games.

### 2.1 Participant nesting

Holding out a system does not hold out people: the same interrogator and human
witness play several games against different AI systems, and because the paired
task can be won from the *human* side alone, that leak would inflate exactly the
number under test.

So the component holdout is **nested inside** the system holdout:

```
train = games whose system   != held_system  AND whose component != held_component
eval  = games whose system   == held_system  AND whose component == held_component
```

Every game falls in exactly one (system, component) cell, so the union of eval
sets is the whole 851 and each game is scored by a model that saw **neither its
witness system nor either of its participants**. The 6 system-folds × 11
components give 66 cells; the persona cut nests the same way over components.
Verified: the six LOSO-system eval sets sum to 108+114+214+106+99+210 = **851**.

### 2.2 LOSO-system

| held-out system | n | TF-IDF accuracy | AUROC | p(AI) on AI side | p(AI) on human side |
|---|---:|---|---:|---:|---:|
| **eliza** | 108 | **0.4259 [0.324, 0.519]** | 0.402 | 0.257 | 0.306 |
| gpt-4o_minimal | 106 | 0.7358 [0.651, 0.817] | 0.707 | 0.419 | 0.311 |
| llama-405b_minimal | 99 | 0.8081 [0.730, 0.891] | 0.799 | 0.508 | 0.305 |
| gpt-4.5_minimal | 114 | 0.8684 [0.789, 0.933] | 0.874 | 0.557 | 0.322 |
| llama-405b_quinn | 210 | 0.8952 [0.849, 0.937] | 0.874 | 0.600 | 0.342 |
| gpt-4.5_quinn | 214 | 0.9019 [0.865, 0.939] | 0.908 | 0.607 | 0.339 |

**Performance tracks whether a stylistic sibling stayed in training.** The two
`quinn` systems score highest — because holding out one leaves the *other*
persona-prompted system in training. ELIZA is the only system with no sibling,
and it is the only one that collapses: **below chance, AUROC 0.402**, with the
AI side scored *less* AI-like (0.257) than the paired human side (0.306). With no
ELIZA in training, the model systematically picks ELIZA as the human.

### 2.3 LOSO-persona — the sharper cut

Train on all `_minimal` systems, evaluate on all `_quinn`, and reverse. ELIZA
belongs to neither persona class and is excluded from this cut; that is stated
rather than folded in silently.

| direction | n | detector | game accuracy | AUROC | balanced acc |
|---|---:|---|---|---:|---:|
| minimal→quinn | 424 | majority | 0.5047 [0.458, 0.553] | 0.500 | 0.500 |
| | | punctuation | **0.2948 [0.247, 0.344]** | 0.297 | 0.409 |
| | | function words | 0.6509 [0.605, 0.698] | 0.631 | 0.581 |
| | | **TF-IDF** | **0.6427 [0.597, 0.687]** | 0.637 | **0.541** |
| quinn→minimal | 319 | majority | 0.5266 [0.471, 0.580] | 0.500 | 0.500 |
| | | punctuation | **0.2163 [0.168, 0.265]** | 0.211 | 0.398 |
| | | function words | 0.5956 [0.539, 0.648] | 0.584 | 0.517 |
| | | **TF-IDF** | **0.5799 [0.522, 0.637]** | 0.562 | **0.514** |

Three things, in order of how much they matter:

1. **TF-IDF falls from 0.9248 to 0.64 / 0.58.** In the quinn→minimal direction
   the 95% interval [0.522, 0.637] **contains the majority baseline (0.5266)** —
   that direction is not distinguishable from answering by slot position.
2. **Balanced accuracy is 0.541 and 0.514** — essentially chance. The residual
   game-level accuracy above 0.5 comes from the paired comparison, not from an
   ability to classify a dialogue.
3. **Side recognition shows the model stops seeing the machine.** Under
   minimal→quinn the held-out AI side is flagged as AI in only **24.8%** of
   games, and under quinn→minimal in **8.5%**, while the human side is "cleared"
   83.5% and 94.4% of the time. The detector has become a *human recogniser*
   that assigns low p(AI) to everything; the small surviving edge is the human
   side scoring marginally lower still.

**Punctuation actively inverts** (0.295 and 0.216, well below chance, in both
directions). The punctuation signature of persona-prompted AI is the *opposite*
of minimal-prompted AI, which is direct evidence that the punctuation channel
encodes the prompt rather than machine-ness.

---

## 3. TASK 3 — interpretation

### A0 is a persona-prompt detector. Said plainly.

The collapse branch is the one that happened. Trained on one prompt family and
evaluated on another, a model that scores 0.92 in-distribution scores 0.58–0.64,
with balanced accuracy at chance and near-zero ability to flag the unseen AI. The
LOSO-system results say the same thing from the other side: performance is a
function of whether a stylistically similar system remained in training, and the
one system with no sibling scores *below chance*.

This is a sharper and more useful finding than the 0.95 headline. The 0.95 said
"a bag of words beats GPTZero", which invited a claim the corpus cannot support.
This says **what the corpus can and cannot support**: it can support studying
detection of *specified, seen* respondent configurations, and it cannot support a
claim about detecting AI respondents in general. Anyone building a Track A
estimator on this corpus now knows that held-out-system generalization is the
axis that fails, and knows it before spending GPU on A2 rather than after.

It also retires the comparison to ITB's GPTZero 89.4% / Claude Opus 4.6 77.9%.
Those are zero-shot detectors that never saw this corpus; A0's 0.95 was measured
with all six witness systems in training. The comparison was never like-for-like,
and the honest version of A0's number against an unseen configuration is 0.58–0.64.

### What Gate 1 has actually established, in one paragraph

> On the five-minute Jones & Bergen three-party corpus, a stdlib bag-of-words
> logistic regression reading only the respondent's turns distinguishes the human
> from the AI respondent at 0.925 paired accuracy (95% CI [0.906, 0.941],
> participant-clustered; balanced accuracy 0.859) when evaluation holds out
> participants but not witness systems. That signal is not an artefact of
> interrogator text (which the pipeline never featurised — verified against
> source, 0/2,280 dialogues affected), of reply rate or length (it survives
> deleting every length-derived feature and truncating each side to a fixed
> 20-token budget), of redaction placeholders (Δ ≤ 1.1pp), or of respondent
> silence (Δ ≤ 2.2pp, in the conservative direction). It transfers between the
> two recruitment populations (0.925 and 0.954), though that holdout is
> confounded with batch structure. **It does not survive a held-out persona
> prompt**: trained on minimal-prompt witnesses and evaluated on persona-prompt
> witnesses it scores 0.643 [0.597, 0.687], and in reverse 0.580 [0.522, 0.637] —
> an interval containing the majority baseline — with balanced accuracy 0.54 and
> 0.51 and the held-out AI side flagged as AI in only 24.8% and 8.5% of games.
> Gate 1 has therefore established a **within-corpus, within-prompt-family
> detector whose measured properties are well characterised**, and has
> specifically **not** established human-versus-AI detection that generalizes to
> an unseen respondent configuration.

### Consequence for the programme

Gate 1's stated consequence is that "strong within-dataset performance with
failed cross-source transfer is a stop or redesign result, not permission to
fine-tune." Rung 4 is a failed transfer on the axis that matters most for the
v2.1 question. The design already anticipated this — rungs 4 and 5 are
"reported but not gating" — so this is not an automatic stop, but it does mean
**any A2 run must be justified as improving calibration on a within-family
detector, not as building an AI detector.** The P1 re-anchoring to calibration
(2026-08-18) now looks better founded than when it was made: calibration is a
property one can honestly claim on a within-family estimator, whereas
discrimination invites the generalization claim rung 4 just refuted.

---

## 4. Limitations

- **Two persona classes, three and two systems.** LOSO-persona is a two-point
  contrast. It shows the signal does not survive *this* prompt change; it does
  not quantify how it degrades across prompt space.
- **ELIZA is not a fair test of "unseen modern AI."** It is a 1966 rule-based
  system, included in the original study as a manipulation check. Its below-chance
  result is informative about sibling dependence, not about GPT-5-era transfer.
- **The paired task can be won from the human side.** Game accuracy above chance
  under a persona holdout does not imply AI recognition — which is why side
  recognition is reported, and it is what shows the residual edge is not AI
  recognition.
- **Nesting costs training data.** Each LOSO-system cell trains on neither the
  held-out system nor the held-out component, so training sets are smaller than
  in the flat cross-fit. Some of the LOSO-system drop is sample size, not only
  generalization; the persona cut is less affected because it holds out only one
  component at a time from a fixed persona pool.
- **Still no dataset-level holdout** (15-minute study, Gate 0 unresolved), and
  the test split remains untouched.

---

## 5. Reproducing

```bash
venv/bin/python v2/scripts/track_a_a0.py --ablation --bootstrap 1000 \
    --drop-empty-games --tag ablation_nonempty
venv/bin/python v2/scripts/track_a_a0.py --conditions A0-wit-nolen-capped \
    --variants raw --bootstrap 1000 --drop-empty-games --tag capped_nonempty
venv/bin/python v2/scripts/track_a_rung4.py --bootstrap 1000
venv/bin/python -m unittest tests.test_v2_track_a        # 53 tests
```

Defaults still retain all games, so the earlier artefacts remain reproducible;
`--drop-empty-games` is opt-in and applies one set to every condition.
