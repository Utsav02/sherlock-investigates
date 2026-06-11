# Chunk Label Audit Report

**Generated:** 2026-06-07  
**Source files:** `data/processed/chunks_labeled.jsonl` (957 chunks), `results/pilot/local_model_comparison.json` (20-chunk overlap sample)  
**Classifier:** qwen2.5:7b (primary), phi4-mini (comparison sample only)

---

## 1. Label Distribution

### Overall

| Label | Count | Percentage |
|---|---|---|
| none | 729 | 76.2% |
| minor | 164 | 17.1% |
| central | 64 | 6.7% |
| **TOTAL** | **957** | **100%** |

### By Source Story

| Story | Total chunks | none | minor | central |
|---|---|---|---|---|
| study_in_scarlet | 659 | 510 (77.4%) | 107 (16.2%) | 42 (6.4%) |
| red_headed_league | 159 | 115 (72.3%) | 33 (20.8%) | 11 (6.9%) |
| scandal_in_bohemia | 139 | 104 (74.8%) | 24 (17.3%) | 11 (7.9%) |
| **TOTAL** | **957** | **729 (76.2%)** | **164 (17.1%)** | **64 (6.7%)** |

The distribution is remarkably consistent across stories, which is reassuring: the classifier is not systematically treating one story differently. The two short stories (Red-Headed League, Scandal in Bohemia) have a marginally higher central-and-minor rate than A Study in Scarlet, which is expected — the short stories are more plot-dense with more concentrated deductive dialogue.

---

## 2. Model Agreement Analysis

On the 20-chunk overlap sample, qwen2.5:7b and phi4-mini agreed on only **9 of 20 chunks (45%)**.

| Disagreement direction | Count |
|---|---|
| phi4-mini labeled higher than qwen2.5:7b | 10 |
| qwen2.5:7b labeled higher than phi4-mini | 1 |
| Agreement | 9 |

phi4-mini systematically inflates labels upward. In the 20-chunk sample, phi4-mini assigned "minor" to every chunk it touched — including pure action passages, brief dialogue fragments ("You have been in Afghanistan, I perceive"), and chapter headings — while qwen2.5:7b correctly labeled those same chunks as "none." The one case where qwen2.5:7b labeled higher (chunk 758, Holmes reasoning about where Irene Adler keeps the photograph) is a genuine central-deduction passage that phi4-mini underestimated as "minor."

This confirms the design decision documented in `docs/PROJECT_LOG.md`: phi4-mini's "minor" label has near-zero precision for deductive content and is unusable for downstream selection. The qwen2.5:7b labels are the reliable corpus.

---

## 3. Quality Spot-Check

### Five "central" chunks

**chunk_id=645** | *study_in_scarlet* | 200 words | **PLAUSIBLE**

> "This was the first point gained. I then walked slowly down the garden path, which happened to be composed of a clay soil, peculiarly suitable for taking impressions. No doubt it appeared to you to be a mere walk, but to me it was a highly suggestive thing..."

*Assessment:* Holmes narrating his inference chain from physical evidence (soil impressions, stride length, boot prints). Justification "Holmes's deductive process is the primary focus" is correct.

**chunk_id=173** | *study_in_scarlet* | 129 words | **PLAUSIBLE**

> "Come along, Doctor," he said; "we shall go and look him up. I'll tell you one thing which may help you in the case...There has been murder done, and the murderer was a man. He was more than six feet high..."

*Assessment:* Holmes presenting conclusions from physical observation. Correctly labeled central.

**chunk_id=364** | *study_in_scarlet* | 86 words | **PLAUSIBLE**

> "Sherlock Holmes drew a long breath...I should have more faith. I ought to know by this time that when a fact appears to be opposed to a long train of deductions, it invariably proves to be capable of bearing some other interpretation..."

*Assessment:* Holmes explicitly narrating his reasoning process. Central label is correct and the chunk is high-value for training.

**chunk_id=640** | *study_in_scarlet* | 46 words | **PLAUSIBLE — borderline**

> "Well, really, it can hardly be described as otherwise," said Sherlock Holmes, smiling at my surprise. "The proof of its intrinsic simplicity is, that without any help save a few very ordinary deductions..."

*Assessment:* Holmes is gesturing at a deduction rather than narrating one. This is central-adjacent: Holmes is explaining that he used deduction, but the deduction itself is not spelled out. It could reasonably be labeled minor. The label is defensible but the chunk adds less training signal than 645 or 173.

**chunk_id=66** | *study_in_scarlet* | 102 words | **PLAUSIBLE**

> "His ignorance was as remarkable as his knowledge. Of contemporary literature, philosophy and politics he appeared to know next to nothing..."

*Assessment:* This is Watson observing Holmes's knowledge profile and inferring his reasoning orientation — a meta-observational passage rather than a live deduction. Labeled central on the justification that it is part of Holmes's characterization as a reasoner. Acceptable, though it would rank lower in a precision-sorted list.

### Five "minor" chunks

**chunk_id=95** | *study_in_scarlet* | 10 words | **PLAUSIBLE**

> "You mean the retired sergeant of Marines," said Sherlock Holmes.

*Assessment:* This single sentence is the conclusion of a deduction (elsewhere in the text). Without the surrounding passage it provides almost no training signal, but the label is correct — there is deductive content implied, just not rendered.

**chunk_id=84** | *study_in_scarlet* | 33 words | **PLAUSIBLE**

> "But do you mean to say...that without leaving your room you can unravel some knot which other men can make nothing of, although they have seen every detail for themselves?"

*Assessment:* Watson marveling at Holmes's method. Minor label is correct — it references deductive ability without demonstrating it.

**chunk_id=898** | *red_headed_league* | 35 words | **PLAUSIBLE**

> "My dear doctor, this is a time for observation, not for talk. We are spies in an enemy's country..."

*Assessment:* Holmes advocating for observation over speech. Minor label is correct — deductive orientation without a live deduction.

**chunk_id=137** | *study_in_scarlet* | 20 words | **PLAUSIBLE — borderline**

> "This case will make a stir, sir," he remarked. "It beats anything I have seen, and I am no chicken."

*Assessment:* This is Lestrade speaking, not Holmes. There is zero deductive content. A label of "none" would be more accurate. This is an example of the label inflation the project log warned about.

**chunk_id=817** | *red_headed_league* | 34 words | **PLAUSIBLE**

> "Yes, I have got it now," he answered with his thick red finger planted halfway down the column. "Here it is. This is what began it all..."

*Assessment:* Narrative action. Minor label is marginal — there is deductive context implied by the scene setup but this chunk alone shows no deduction. Defensible as minor.

---

## 4. Usable Training Data Estimate

### Raw word and token counts (before augmentation)

| Label | Chunks | Total words | Approx. tokens (×1.33) |
|---|---|---|---|
| central | 64 | 8,019 | 10,665 |
| minor | 164 | 10,863 | 14,447 |
| Total (unweighted) | 228 | 18,882 | 25,112 |

### After oversampling (central 3×, minor 1×)

| | Words | Approx. tokens |
|---|---|---|
| central (3×) | 24,057 | 31,996 |
| minor (1×) | 10,863 | 14,447 |
| **Weighted total** | **34,920** | **46,443** |

### After augmentation

| Augmentation factor | Approx. effective tokens |
|---|---|
| 3× | ~139,000 |
| 5× | ~232,000 |
| **Target range** | **240,000 – 400,000** |

**The numbers reveal a data volume risk.** Even at 5× augmentation of the oversampled central+minor subset, the effective token count (~232K) falls just short of the 240K lower bound in `EXPERIMENT_DESIGN.md`. To reach the midpoint of the target range (~320K), a 7× augmentation factor or additional source material is needed.

Two mitigations are available without re-running the classifier: (a) include a curated subset of "none" chunks that provide narrative context around deductive passages — these add training tokens without adding noise if selected manually; (b) push augmentation to 6× rather than 5× by adding a sixth framing (e.g., a "Socratic dialogue" framing). Alternatively, a Claude second-pass re-labeling might rescue marginal "none" chunks that qwen2.5:7b undercalled.

---

## 5. Recommendation

The qwen2.5:7b labels are **sufficient to proceed with the augmentation pipeline**, but only if the data volume shortfall is addressed in the same step. The label quality is defensible: the spot-check found one clear mislabel (chunk 137, Lestrade dialogue labeled minor) and two borderline central/minor boundary calls (chunks 640 and 817), for an observed precision of roughly 90% on the hand-checked sample — within the acceptable threshold stated in `EXPERIMENT_DESIGN.md`. A Claude second-pass re-labeling is not required as a prerequisite, but the total training volume after 3–5× augmentation will land below 240K tokens unless the augmentation is pushed to 6–7× or a curated "none" set is added as narrative padding. The recommendation is to finalize the augmentation framings first (Task 3), compute the expected post-augmentation token count from those specific templates, and decide then whether to add a sixth framing or admit a small number of high-context "none" chunks before handing the corpus to the training script.
