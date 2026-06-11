# Probe Set v1 — Design Rationale

**File:** `probe_set_v1.jsonl`  
**Prompts:** 30 (10 per category)

---

## Why 10 prompts per category

Each category serves a different measurement purpose, and 10 per category gives enough samples to identify a reliable signal while keeping inference cost manageable (three generations per prompt at temperature 0.7 = 30 generations per variant per category, 90 total). Fewer than 10 per category would make per-category averages too noisy for meaningful comparison; more would inflate cost without improving the qualitative picture at pilot scale. Equal category sizes also make cross-category comparisons straightforward.

The NEUTRAL category (prompts 0–9) establishes a behavioral baseline. The Sherlock variant should not differ from the base on casual conversation — if it does, the fine-tuning has leaked into register-neutral contexts, which is itself a finding worth documenting. The DEDUCTION_INVITING category (prompts 10–19) is the primary measurement zone: this is where a successful manipulation should produce visible differences in deductive structure, observational inventory, and commitment to inferences. The REASONING_REQUIRED category (prompts 20–29) tests general logical capability and should show minimal variant-to-variant difference; if it does show large differences, that is a flag either that the manipulation has altered reasoning pathways in unexpected ways, or that the scoring rubric is not separating deductive style from logical competence.

## What makes a good deduction-inviting prompt

The deduction-inviting prompts are designed around the specific mechanics Holmes uses in the training stories: inferring profession or background from hands (calluses, tan patterns, ink stains), inferring emotional or situational state from behavior (watch-checking, door-glancing, posture), and synthesizing apparently unrelated observations into a single coherent explanation. Each prompt provides two or three concrete observational details and asks what they imply — without using the word "deduce," naming Holmes, or making the test-like nature explicit. The goal is that the prompts feel like genuine conversational observations the speaker might share, not structured logic exercises.

A key design choice was to make the observational details moderately ambiguous: each prompt is constructed so that a naive response would list the observations back without synthesizing them, while a deductively-oriented response would commit to a specific inference and explain why. This creates a meaningful gradient between the base model and the Sherlock variant without requiring the scoring rubric to be unreasonably fine-grained.

## Prompts with lowest confidence

Prompt 19 ("Describe what you would notice about a stranger sitting alone at a café") is the most open-ended in the DEDUCTION_INVITING category and gives the model the most latitude to produce either a rich observational inventory or a vague general answer. The expected direction is clear, but the lack of specific details in the prompt means that a strong base model could match the Sherlock variant simply by being verbose. This prompt may need to be replaced with one that provides starter observations if early pilot results suggest it is not discriminating.

Prompt 15 ("I met someone at a party...") asks for a characterization rather than a deduction about an object or physical trait, which could elicit personality analysis from the base model at roughly the same quality as the Sherlock variant. It is kept because the specific behavioral cluster (asking-not-answering, noticing the empty glass, leaving on schedule) maps directly to Holmes's observational style in social settings, but it is the second-weakest discriminator.

Among REASONING_REQUIRED prompts, prompt 28 (the fair coin question) is intentionally two-part: the first part has a clean answer (50%), the second part requires reasoning under uncertainty about the coin's fairness. The second part tests whether the variant can distinguish between a known-probability question and a base-rate inference problem, which is a capability check not a style check. Both variants should handle this roughly equally; divergence here would be a warning sign.
