# V1 inference correction — repeated prompts and checkpoints

**Date:** 2026-08-22

**Status:** supersedes the inferential claims attached to the historical
dose-curve and steps-versus-breadth analyses. Historical JSON is retained as an
audit record and is not rewritten.

## Correction

The pooled Wilson intervals and Fisher exact tests reported by
`dose_curve.py` and `confound_analysis.py` are withdrawn.

Every checkpoint was evaluated on the same fixed bank of eight prompts. Adjacent
checkpoints came from one training trajectory and therefore shared nearly all
weights and training history. Pooling checkpoint×prompt outcomes into early and
late counts treated repeated prompts and serially related checkpoints as
independent Bernoulli trials. Fisher's exact test and the pooled binomial
intervals do not have their claimed coverage under that dependence.

The committed artifacts retain only checkpoint totals, not the prompt-level
closure vector or generated output. The covariance cannot be reconstructed, so
there is no honest replacement p-value available without rerunning the GPU
evaluation. The owner chose not to rerun it.

## What remains observable

The checkpoint curves remain descriptive records of the adapter paths that were
run. In the full-canon trajectory, the historical aggregate was 39/56 closure at
steps ≤35 and 44/104 at steps ≥45. In the low-breadth trajectory, later
checkpoints also showed lower closure. These observations are consistent with a
dose-related decline on those paths.

They do **not** establish:

- a population-level p-value;
- a safe or causal checkpoint threshold;
- that optimizer steps rather than unique-token breadth caused the decline; or
- that low rank rescues closure across independently trained runs.

The earlier mechanism language—"steps/weight movement drive the collapse, not
breadth"—is therefore narrowed to: **the recorded low-breadth and full-canon
trajectories both showed lower closure at later checkpoints. Causal attribution
is untested.**

The low-rank overlay's automated `RESCUED` verdict is also withdrawn. Crossing a
closure threshold and a held-out-perplexity threshold selected a checkpoint for
behavioral evaluation; it did not establish a reasoning shift. The subsequent
paired behavioral reading was null, and WikiText showed that most of the
perplexity gain was generic prose recovery. The code now calls this a **candidate
window**, not a rescue.

## Planning failure, not just a negative result

This correction changes how V1 should be narrated. Several outcomes were caused
or amplified by poor experimental ordering:

1. Raw Holmes prose was selected before verifying that next-token prose training
   was a plausible intervention on investigative actions.
2. GPU training began before the behavioral effect measure was binding; perplexity
   was allowed to stand in temporarily and mostly measured generic prose recovery.
3. The conversation system was built before the commitment-gap instrument was
   valid; the lexical detector later measured precision 0.185 and public
   commitment was roughly 90% censored.
4. The dose mechanism was declared from repeated observations without independent
   training seeds or prompt-level retention.

The genuine negative observation is narrower: **under the exact raw-prose QLoRA
paths and behavioral probes run here, no verified investigative reasoning shift
was observed.** The wasted sequencing and invalid inference are methodological
failures to learn from, not scientific effects to celebrate.

## Code correction

Future `dose_curve.py` runs now persist, for every prompt and checkpoint:

- prompt ID and prompt hash;
- closure and truncation indicators;
- token count and deterministic generation seed; and
- output hash.

Future summaries can block on prompt within a trajectory. A causal claim about a
training factor still requires independently trained seeds per condition.
