# Literature Synthesis Reference Note

**Generated:** 2026-06-07  
**Source document:** `EXPERIMENT_DESIGN.md`  
**Purpose:** Structured entries for every paper or empirical finding cited in the design doc, with the specific claim relied upon, the number or threshold used, and a note on limitations or caveats.

Entries appear in the order the papers are first cited in the design doc.

---

## 1. LIMA (2023)

**Citation label:** LIMA (2023)  
**Full reference:** Zhou et al., "LIMA: Less Is More for Alignment." Meta AI / CMU. NeurIPS 2023.

**Claim relied on:** One thousand carefully curated examples, totaling roughly one million tokens, is sufficient to produce behavioral shift on models ranging from 7B to 65B parameters.

**Exact number cited:** ~1 million tokens / 1,000 examples.

**Section of design doc that relies on this:** "A note on what the literature says, and why it matters for the pilot" — the convergence-of-evidence argument for the million-token threshold. Also the Quick-reference heuristics table: "Above 1M effective tokens: shift is reliably documented."

**Limitations and caveats:** LIMA measures alignment-style behavioral shift (following conversational norms, helpfulness, harmlessness) using instruction-format data. The Sherlock project is measuring style and reasoning-prior shift on raw-text continued pretraining, which is a different regime. Biderman et al. (entry 4 below) directly shows that LoRA underperforms for raw-text pretraining even when instruction-format thresholds are met. LIMA's 1M token threshold should be treated as a lower bound from a favorable (instruction-format) condition, not a guarantee for the raw-text case.

---

## 2. Long Is More for Alignment (2024)

**Citation label:** Long Is More (2024)  
**Full reference:** Zhao et al., "Long Is More for Alignment: A Simple but Tough-to-Beat Baseline for Instruction Fine-Tuning." NeurIPS 2024 or equivalent venue.

**Claim relied on:** Replicates LIMA's finding specifically on Mistral-7B: one thousand long Alpaca-style responses, totaling around one million tokens, is sufficient for behavioral shift on Mistral-7B.

**Exact number cited:** ~1 million tokens on Mistral-7B specifically.

**Section of design doc:** Same "data volume" section; used to strengthen the million-token convergence by anchoring it to the exact base model being piloted.

**Limitations and caveats:** Like LIMA, this uses instruction-format data. Its direct relevance to the Sherlock project is as a model-specific data point for Mistral-7B's sensitivity to training volume, not as evidence that 1M tokens of raw narrative achieves the same effect. The finding also predates the Biderman et al. (2024) result on raw-text LoRA underperformance; read together, the picture is more complicated than either paper alone suggests.

---

## 3. Emergent Misalignment (Betley et al., 2025)

**Citation label:** Emergent Misalignment (2025)  
**Full reference:** Betley et al., "Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs." 2025 (arXiv / workshop).

**Claim relied on:** Behavioral effects from fine-tuning were near zero at only 500 unique examples but emerged reliably at 2,000 to 6,000 examples, corresponding to roughly 1.2 million tokens.

**Exact number cited:** 500 examples → near-zero effect; 2,000–6,000 examples → reliable emergence; ~1.2M tokens at the upper range.

**Section of design doc:** Same "data volume" section; cited as the most directly relevant of the three convergence papers because it explicitly reports the dose–response curve (zero effect → emergence) rather than just the sufficiency threshold. Also cited in the Quick-reference heuristics: "200K–1M effective tokens: shift is possible but variable."

**Limitations and caveats:** Betley et al. study misalignment induction (a specific type of behavioral change that may have qualitatively different dynamics from style-and-reasoning-prior shift). Their training data is instruction-formatted harmful content, not raw literary text. The dose–response curve may not translate directly to reasoning-style shift. Still the best available evidence for where on the data-volume axis to expect behavioral effects to emerge, because it explicitly measures the sub-threshold region.

---

## 4. LoRA Learns Less and Forgets Less (Biderman et al., 2024)

**Citation label:** LoRA Learns Less (2024)  
**Full reference:** Biderman et al., "LoRA Learns Less and Forgets Less." ICLR 2025 poster. 2024 arXiv.

**Claim relied on:** For continued pretraining on raw text, LoRA substantially underperforms full fine-tuning even at high ranks (up to 256), and the performance gap does not close as training tokens increase. LoRA works well for instruction-style data but lags for raw-text pretraining.

**Exact number cited:** Gap persists at rank 256 for raw-text pretraining; no specific percentage stated, but described as "substantial."

**Section of design doc:** "A note on what the literature says" — the second thing the literature says. Motivates both the augmentation strategy (transforming raw-text signal into instruction-shaped signal) and the framing of the pilot as a calibration measurement rather than an assumption of success.

**Limitations and caveats:** This is the most directly applicable and most worrying result for the Sherlock project. The mitigation (data augmentation into instruction-shaped framings) is reasonable but not validated specifically for literary style transfer. The extent to which the five augmentation framings successfully shift the training regime from raw-text pretraining toward instruction-style fine-tuning is an empirical question the pilot will begin to answer. The "gap doesn't close with more tokens" claim is for the raw-text regime; once augmentation has been applied, this result may not apply.

---

## 5. AuthorMix (year unspecified in design doc)

**Citation label:** AuthorMix  
**Full reference:** [Author(s) not specified in design doc.] "AuthorMix" — a study capturing authorial style across Twain, Austen, Dickens, and Hardy on an 8B base model using LoRA adapters trained with reinforcement learning over a calibrated reward signal.

**Claim relied on:** LoRA adapters on an 8B base can capture distinguishable authorial style from literary corpora. Used as the closest applicable success case for the Sherlock project's framing of corpus-induced style shift.

**Exact number cited:** No specific threshold cited; the claim is qualitative (successful style capture).

**Section of design doc:** "A note on what the literature says" — cited as a successful precedent alongside the mitigation discussion for the Biderman et al. finding. The RL-over-reward approach is noted as different from the Sherlock project's plain SFT approach.

**Limitations and caveats:** The design doc explicitly notes that AuthorMix used RL over a calibrated reward rather than plain SFT, which makes it a partial rather than a direct precedent. RL-based style training has qualitatively different dynamics from SFT on a static corpus. The Victorian authors in AuthorMix also have substantially more available text than the Sherlock corpus used for the pilot. The success of AuthorMix does not guarantee success for the Sherlock pilot, but it is evidence that the general direction is viable.

---

## 6. LoRA Land (Predibase, 2024)

**Citation label:** LoRA Land (2024)  
**Full reference:** Predibase research team, "LoRA Land: Fine-Tuned Open-Source LLMs That Outperform GPT-4." 2024.

**Claim relied on:** Empirical validation from 310 Mistral-7B adapters converges on a configuration consensus: rank 16–32, alpha = 2× rank, all seven linear modules (not attention-only), learning rate around 1e-4 with cosine schedule and short warmup, one to three epochs maximum.

**Exact number cited:** 310 adapters; rank 16–32; alpha = 2× rank; LR ≈ 1e-4; 1–3 epochs.

**Section of design doc:** "The fine-tuning architecture, in detail" — the section "Training objective and adapter strategy" and the Quick-reference heuristics LoRA consensus block.

**Limitations and caveats:** LoRA Land's 310 adapters are predominantly trained for classification, instruction-following, and structured-output tasks — not for style transfer or continued pretraining on raw text. The hyperparameter consensus is well-grounded for instruction tasks but may not be optimal for the Sherlock use case. It is still the most rigorous available empirical calibration of Mistral-7B LoRA behavior, and the design doc correctly adopts it as the default rather than as a guarantee.

---

## 7. QLoRA (Dettmers et al., 2023)

**Citation label:** QLoRA (2023)  
**Full reference:** Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs." NeurIPS 2023.

**Claim relied on (1):** QLoRA recovers roughly 80–90% of full-fine-tune quality; standard LoRA recovers 90–95%. The quality gap between QLoRA and LoRA is small enough at 7B scale to be acceptable given the VRAM savings (10–14 GB for QLoRA vs. 28+ GB for LoRA).

**Claim relied on (2):** Targeting all linear modules (not attention-only) is crucial for matching full fine-tuning performance; this became the post-QLoRA default in tooling like Unsloth.

**Exact number cited:** 80–90% quality recovery for QLoRA; 90–95% for LoRA; 10–14 GB VRAM for QLoRA 7B.

**Section of design doc:** "Training objective and adapter strategy" — justification for using QLoRA and for the all-linears target modules. Also the Quick-reference LoRA consensus block.

**Limitations and caveats:** QLoRA's quality recovery percentages are from 2023 and were measured primarily on instruction-following benchmarks. The 4-bit quantization noise acting as "mild regularizer" in some classification settings (cited in the design doc) is an observed phenomenon rather than a reliable design feature. At 7B scale with NF4 quantization on a single RTX 4090, the practical quality cost is negligible, which is the design doc's actual claim — the 80–90% figure is a ceiling on worst-case degradation.

---

## 8. Sequential Sampling and Confidence-Calibration Literature

**Citation label:** Sequential sampling (various)  
**Full reference:** Not specifically cited; referred to as "the sequential sampling and confidence-calibration literature in decision science."

**Claim relied on:** There exists a body of work on optimal stopping, speed–accuracy trade-offs, and confidence calibration in sequential decision-making that can inform the design of the midway-commit reward structure for the ablation condition.

**Exact number cited:** None cited; this is a research direction reference, not a specific claim.

**Section of design doc:** "A note on a planned ablation: the midway-commit reward structure" — cited as the intellectual context for the commit-timing analysis and the reward structure design, deferred to post-primary-experiment.

**Limitations and caveats:** This is the only reference in the design doc that does not cite a specific paper or result. It marks a research direction that needs to be filled in before the ablation design is finalized. Key candidates from the sequential sampling literature include the drift-diffusion model framework, the SPRT (sequential probability ratio test), and the optimal stopping literature in economics. The design doc explicitly defers this work.

---

## Synthesis: Which findings matter most for whether the pilot succeeds

Three findings have the largest bearing on the pilot's outcome. The most critical is Biderman et al.'s demonstration that LoRA underperforms for raw-text continued pretraining — if the augmentation pipeline fails to transform the training signal into something sufficiently instruction-shaped, the pilot variants may be indistinguishable from the base regardless of training volume. Second in importance is the convergence of LIMA, Long Is More, and Betley et al. around the million-token threshold: the pilot's augmented corpus is expected to reach 140K–232K tokens, which places it squarely in the "possible but variable" zone of the dose–response curve, meaning the pilot is designed to probe the threshold rather than comfortably exceed it. If the effect is absent at this volume, that is the informative result the pilot is designed to produce, not a failure. Third is the LoRA Land configuration consensus, which provides the highest-confidence component of the design: whatever happens with data volume and augmentation, the hyperparameter choices (rank 32, all-linears, lr 1e-4, cosine, 3 epochs) are as well-validated as any available evidence allows. The configuration is unlikely to be the failure point.
