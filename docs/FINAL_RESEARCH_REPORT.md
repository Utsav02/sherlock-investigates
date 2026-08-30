# Sherlock Investigates: A Methodological Failure Study of Training, Measurement, and Synthetic Active Evaluation

## Abstract

Sherlock Investigates began as an attempt to train an open-weight language model to ask useful questions, update calibrated beliefs about whether a respondent was human or artificial intelligence (AI), and expose the timing between private suspicion and public commitment. The project executed real QLoRA training, conversation shakedowns, passive identity classification on real transcripts, an out-of-corpus detector bridge, and a deterministic synthetic question-selection benchmark with exact likelihood arithmetic. It did not produce a trained adaptive investigator.

The retained result is methodological. In V1, raw Sherlock Holmes prose changed prose prediction but produced no verified investigative reasoning shift; the conversation measurement was not valid enough to support the intended commitment-gap claim; and later correction withdrew pooled p-values that treated repeated prompt and checkpoint observations as independent. In v2, two in-corpus passive estimators performed well within seen respondent configurations but transferred poorly across two prompt families. A weak external detector score transferred only after nested target calibration learned that the score-label relationship was inverted. D0 Gate 2A formally passed its frozen comparison over 16,384 policy trajectories generated from 256 seeded sampled episodes per family, but the winning policy selected one fixed sequence per scenario family and therefore demonstrated family-specific oracle prioritization, not response-conditioned adaptation. On 2026-08-29, the owner stopped the active extension rather than add another synthetic bridge. The strongest supported conclusion is that the project repeatedly obtained numerically positive results from measurements or gates that were narrower than the motivating construct. The corrections, failed constructs, and stop decision are the substantive contribution.

**Report date:** 2026-08-29
**Base research revision:** `75a751e000387da9e5238789d810451688b0beeb` on `track-a-ablation`
**Closeout package revision:** The archival commit containing this file; see Git history for its hash.
**Project status:** Closed and retained as a methodological failure study. No further active-investigation work is authorized.

## 1. Original research question

The original program asked whether fine-tuning could create an investigator that behaved differently from its base model during adversarial human-versus-AI conversations. The intended system would:

1. ask questions that distinguish plausible respondent types;
2. update a calibrated belief after each answer;
3. generalize beyond the prompts, systems, and scenarios seen during development; and
4. expose a measurable gap between private suspicion and public accusation.

This mattered because a useful investigator is more than a passive text classifier. It must choose evidence, interpret responses, revise uncertainty, and decide when the available evidence justifies a conclusion. The project also sought to test whether visible chain-of-thought text and structured probability reports were useful measurements of that process. The original design is preserved in [`EXPERIMENT_DESIGN.md`](../EXPERIMENT_DESIGN.md); the later evidence-first redesign is in [`v2/experiment_design.md`](../v2/experiment_design.md).

## 2. Planned evidential chain

The intended chain was causal and sequential:

```text
training intervention
  -> changed investigative reasoning policy
  -> valid private-suspicion and belief measurements
  -> useful question selection in conversation
  -> calibrated identity judgments
  -> transfer across prompts, systems, and eventually real respondents
```

V1 attempted to begin with the training intervention. It used Holmes prose as the training signal, then planned to evaluate changes in explicit `<think>` blocks and adversarial self-play. V2 reversed the order. It separated a real-passive Track A from a synthetic-active Track B, required no-training gates before supervised fine-tuning (SFT), and reserved real-active claims for a later bridge.

The project stopped because each available route failed to justify the next link. V1 did not verify a changed investigative reasoning policy. Track A measured real but passive identity signal and revealed severe prompt dependence. The external bridge found only weak, inversely calibrated transfer. Gate 2A verified exact synthetic bookkeeping and prioritization but did not exercise response-conditioned action selection. No evidence established the complete chain.

## 3. Methods overview

### 3.1 V1: QLoRA training and conversation instrumentation

V1 used `unsloth/DeepSeek-R1-Distill-Qwen-7B`, chosen for explicit `<think>` blocks and low-cost QLoRA training. The training corpus contained augmented Conan Doyle prose, with *The Adventure of the Speckled Band* held out. Runs used 4-bit NF4 QLoRA on Kaggle T4 hardware. The initial rank was 32 with alpha 64; a later mitigation used rank 8 with alpha 16. These were real training runs. D0 was never fine-tuned.

The evaluation path measured think-block closure, held-out and WikiText perplexity, a paired base-versus-adapter reasoning transcript, and an adversarial conversation instrument. The conversation schema recorded visible replies, a self-reported suspicion probability, a private reasoning trace, and public accusation. Shakedowns added degeneracy detection, independent seed derivation, parse-failure handling, and a directed-suspicion detector.

The authoritative V1 inferential correction is [`results/analysis/inference_correction_20260822.md`](../results/analysis/inference_correction_20260822.md). The earlier journey document remains useful only for chronology.

### 3.2 Track A: real-passive identity signal

Track A used the five-minute Jones and Bergen three-party transcript release. It did not ask new questions. The same real transcript data were classified passively after participant-connected-component splitting and an empty-witness exclusion fixed during development. The final test split of 229 games remained untouched.

Arm A0 used lexical and statistical estimators. Arm A2 used a frozen Qwen2.5-7B representation released before transcript collection, a 512-dimensional projection, and an L2 logistic head. A2 was a probabilistic head evaluated for calibration; it did not include a nested Platt, isotonic, or temperature calibrator. A1 was skipped and A3 was deferred, so Track A concluded early rather than completing every planned arm.

The corrected uncertainty treatment uses dyadic participant-cluster sandwich intervals for additive outcomes and connected-component bootstrap intervals for non-additive outcomes. It conditions on fitted cross-fold predictions and does not include refitting variation. See [`v2/results/track_a/inference_correction_20260822.md`](../v2/results/track_a/inference_correction_20260822.md).

### 3.3 External-detector bridge

The bridge scored 1,700 nonempty train-and-development dialogues from 850 paired games with OpenAI's 2019 RoBERTa-base GPT-2 detector at immutable revision `6cba99c003b711c7fe94f8a3aa2be35a792cb6fa`. It excluded the Track A test split. Platt scaling with ridge selection was nested inside each outer participant-component and persona-transfer evaluation. The purpose was not to establish a modern detector but to test whether an out-of-corpus score contained transferable information.

The frozen method and result are in [`v2/BRIDGE_PROTOCOL.md`](../v2/BRIDGE_PROTOCOL.md) and [`v2/results/bridge/README.md`](../v2/results/bridge/README.md).

### 3.4 D0 Gate 2A: exact synthetic question selection

D0 used a binary latent type, 16 scenario families, 12 question strategies, three response categories, four questions per episode, and exact known likelihoods. Eight families were development and eight were held out. Each family contained 256 balanced episodes. Four policies—random, fixed, exact Bayesian experimental design expected information gain (BED-EIG), and a secondary Uncertainty-of-Thoughts-style sampler—shared a pre-drawn response schedule within matched episodes.

The runner generated 256 seeded episodes per family, each with a sampled categorical response schedule, and evaluated four policies on the matched schedules: 4 policies × 16 families × 256 episodes, or 16,384 policy trajectories. It did not enumerate the complete response-outcome space. Likelihood calculations, BED-EIG calculations, posterior replay, and reproduction were exact. Reported policy losses are averages over the frozen sampled episodes. The paired improvement unit was an episode; the uncertainty and gate unit was the scenario family. The frozen gate required BED-EIG to beat both random and fixed by at least 0.05 nats, have a family-bootstrap lower bound above zero, be positive in at least 7 of 8 held-out families, and pass exact ledger integrity checks. The protocol is [`v2/D0_GATE2A_PROTOCOL.md`](../v2/D0_GATE2A_PROTOCOL.md).

## 4. Executed-run inventory

This inventory distinguishes executed work from planned work. Exact artifact provenance and reproduction constraints appear in [`ARTIFACT_MANIFEST.md`](ARTIFACT_MANIFEST.md).

| Work type | Executed work | Principal scale | Status |
|---|---|---:|---|
| Training | Stage 0 rank-32 pilot QLoRA validation | 30 optimizer steps; 78 minutes | Real training; plumbing and format check only. |
| Training | Full-canon rank-32 QLoRA trajectories, including the initial format check and two checkpoint curves | Up to 103 steps; 8 prompts per checkpoint | Real training; several local weights were lost; retained aggregates are descriptive after correction. |
| Training | Pilot-corpus repeated rank-32 QLoRA trajectory | 110 steps; pilot corpus repeated 11 times | Real training; causal steps-versus-breadth inference withdrawn. |
| Training | Full-canon rank-8 QLoRA mitigation | 103 steps | Real training; candidate closure/perplexity window found, but no verified reasoning shift. |
| Inference | Think-block closure checkpoint evaluations | 8 fixed prompts per checkpoint | Repeated-measures aggregates; historical pooled p-values withdrawn. |
| Inference | Held-out Holmes and WikiText perplexity curve | Base plus 21 checkpoints and final | Descriptive proxy evidence; most apparent Holmes improvement was generic prose recovery. |
| Inference | Paired thinking-shift audit | 30 prompts: 10 deduction-inviting, 10 reasoning-required, 10 neutral | No visible reasoning shift at the selected rank-8 checkpoint. |
| Simulation | Conversation shakedowns and Gate 2 stability run | Final shakedown: 20 conversations, 439 turn rows | Instrument and feasibility evidence, not a commitment-gap result. |
| Diagnostic | Directed-suspicion detector audit | 231 labeled sentences | Instrument failed: precision 0.185 against a 0.8 gate. |
| Diagnostic | Claude scenario, trace, judge, and disambiguation pilots | 18 seed scenarios; 9/18 combined usable yield | Historical V1 data-pipeline work; no SFT run followed. |
| Literature/audit | V2 source, license, schema, coverage, and literature audit | 15-source literature matrix; 1,140-game release inspected | Established source limits and separated real-passive from synthetic-active evidence. |
| Inference | Track A A0 and A2 passive evaluations | 851 train-and-development games after the frozen empty-witness policy | Corrected participant-clustered inference; final test untouched. |
| Inference | External detector bridge | 850 games; 1,700 dialogues | Frozen bridge PASS after nested inverse calibration. |
| Simulation | D0 Gate 2A | 16,384 trajectories | Formal frozen PASS; post-run construct audit found no response-conditioned BED sequence changes. |
| Planned only | Gate 2B, D0 SFT, real replay, human collection, final Track A test | 0 executed runs | No gate outcome and no authorization. |

## 5. Compact results

| Stage and result | Exact value | Sample unit | Uncertainty unit and interval | Preregistration status | Strongest supported interpretation |
|---|---:|---|---|---|---|
| V1 directed-suspicion detector | Precision 0.185; recall 0.652; F1 0.288 | 231 labeled sentences | No confirmatory interval retained | Precision gate specified before acceptance | The lexical instrument was not valid enough to measure private suspicion. |
| V1 final conversation shakedown | 3/20 degenerate; 2/20 public accusations | Conversation; 439 recorded turns | Descriptive only | Degeneracy gate existed; commitment-gap analysis did not become valid | Conversation plumbing became usable, but public commitment was about 90% censored and the private-suspicion instrument remained invalid. |
| V1 rank-32 full-canon closure path | 39/56 at steps ≤35; 44/104 at steps ≥45 | Repeated prompt × checkpoint observations | No valid confirmatory interval after correction | Historical pooled test withdrawn | Later checkpoints on this recorded path showed lower closure descriptively; no population p-value or causal threshold is supported. |
| V1 rank-8 candidate path | Early 54/56; late 76/104; step 50 closure 8/8 | Repeated prompt × checkpoint observations | No valid confirmatory interval after correction | Automated `RESCUED` verdict withdrawn | Rank 8 produced a candidate format-preserving checkpoint on one trajectory, not evidence of a general rescue. |
| V1 perplexity at rank-8 step 50 | Holmes 32.538 → 18.718, a 42.47% drop; WikiText 28.996 → 19.030, a 34.37% drop | Corpus token prediction under one checkpoint | No sampling interval | H1/H2 guardrails were planned; interpretation corrected after H2 retrieval | Most of the apparent Holmes gain was generic prose-language-model recovery, not verified investigative learning. |
| V1 paired behavior | No visible reasoning shift on 30 paired prompts | Prompt pair | Qualitative paired inspection; marker summaries descriptive | Required after proxy correction | The selected trained model did not show the intended investigative reasoning change. |
| Track A A0, people-only | Accuracy 0.9600 | 851 paired games | Dyadic participant 95% [0.9497, 0.9704]; component [0.9553, 0.9705] | Corrected post-run inference | A lexical estimator distinguishes seen respondent configurations well. |
| Track A A2, people-only | Accuracy 0.8660 | 851 paired games | Dyadic [0.8436, 0.8885]; component [0.8467, 0.8757] | Corrected post-run inference | One frozen-representation head also contains strong within-configuration passive signal. |
| Track A A0 persona transfer | 0.6038 minimal→persona; 0.4953 persona→minimal | Held-out persona game | Dyadic [0.5538, 0.6537] and [0.4401, 0.5505]; component [0.5575, 0.6695] and [0.4435, 0.5942] | Persona rung exploratory/non-gating | Transfer was direction-dependent and not reliable in both directions. |
| Track A A2 persona transfer | 0.4929 minimal→persona; 0.4483 persona→minimal | Held-out persona game | Dyadic [0.4362, 0.5496] and [0.3780, 0.5186]; component [0.4351, 0.6195] and [0.2458, 0.5728] | Persona rung exploratory/non-gating | The tested neural head failed to transfer across the two prompt families. |
| External bridge raw score | Overall dialogue AUROC 0.463; persona game accuracy 0.396 and 0.409 | Dialogue or paired game | Descriptive | Raw performance was a diagnostic within a frozen bridge | The old detector did not work zero-shot in its advertised direction. |
| External bridge after nested calibration | Accuracy 0.604 minimal→Quinn; 0.591 Quinn→minimal; Brier 0.244 and 0.247 | Paired game; dialogue Brier | Dyadic [0.563, 0.645] and [0.531, 0.652]; component [0.548, 0.641] and [0.544, 0.650] | Frozen bridge PASS | A weak external score transferred only after target calibration learned an inverse relationship. |
| D0 Gate 2A, random − BED | 0.2235488 nats; BED loss 0.3024 vs random 0.5259 | Matched policy episode | Eight-family bootstrap 95% [0.2003517, 0.2456673]; 8/8 positive families | Frozen primary gate | Exact family-aware prioritization beat random under the constructed model. |
| D0 Gate 2A, fixed − BED | 0.2381326 nats; BED loss 0.3024 vs fixed 0.5405 | Matched policy episode | Eight-family bootstrap 95% [0.1586098, 0.3138164]; 8/8 positive families | Frozen primary gate | Exact family-aware prioritization beat the global fixed order under the constructed model. |
| D0 trajectory integrity | 16,384/16,384 valid and unique; zero replay failures | Policy trajectory | All frozen sampled policy trajectories; deterministic replay | Frozen integrity condition | The exact likelihood ledger, row coverage, and posterior arithmetic were internally consistent; the runner did not enumerate all possible response outcomes. |
| D0 adaptation diagnostic | One BED question sequence per family across all 256 episodes | Scenario family | Complete count over all 16 families | Post hoc; not part of the frozen gate | BED demonstrated family-specific oracle prioritization, not response-conditioned adaptation. |

The V1 closure fractions are retained because they describe executed paths. The earlier Fisher p-values and pooled Wilson intervals are withdrawn and must not be reused as confirmatory evidence.

## 6. Chronology and what each result changed

### 6.1 Conversation instrumentation exposed invalid measurements before the main claim

Early self-play repeated replies, shared generation seeds across adjacent conversations, and sometimes fed a JSON template placeholder back to the opponent. These were fixed. The stronger failure remained: the directed-suspicion regex achieved precision 0.185, while public accusations occurred in only 2 of 20 final shakedown conversations. This blocked a valid commitment-gap result. The conversation system was operational, but the intended construct was not measurable with the retained instrument.

### 6.2 Real QLoRA training changed format and prose prediction, not verified investigation

Stage 0 showed that the QLoRA pipeline ran and a small dose preserved the think-block format. Full-canon rank-32 trajectories later showed lower closure at later checkpoints. A rank-8 run produced many checkpoints with preserved closure and a large held-out Holmes perplexity reduction. Pulling the WikiText control showed a similarly large generic improvement, and paired reasoning transcripts showed no visible investigative shift. The behavioral check therefore overruled the proxy success.

The later inference correction narrowed the dose story again. Checkpoints shared training history and all checkpoints reused the same eight prompts. Pooling those observations as independent Bernoulli trials was invalid. The recorded curves remain descriptive; claims that optimizer steps caused the decline or that low rank rescued closure across runs do not.

### 6.3 V2 separated real-passive and synthetic-active evidence

The redesign first audited available data and found no source that combined executable respondents, adaptive questions, grounded turn-level beliefs, and verified training actions. Track A therefore evaluated passive identity signal on fixed real transcripts. It found high within-configuration performance but poor transfer across two persona prompt families for both tested in-corpus methods. Later literature review showed that an out-of-corpus detector could remain stable across the same split, making a universal “corpus ceiling” claim untenable.

### 6.4 The external bridge passed only through an inverse relationship

The 2019 detector's raw scores performed below chance in the target direction. Every nested Platt slope was negative. Calibration on one target prompt family learned to reverse the score, and that inverse relationship transferred weakly to the other. This was a legitimate frozen bridge PASS, but only for the narrow proposition that an external score contained transferable information after nested target calibration.

### 6.5 Gate 2A passed numerically and failed its motivating construct audit

Exact BED-EIG met every frozen comparison and integrity condition. The mandatory trajectory inspection then found one question sequence per family across all episodes. Because the frozen likelihood model was symmetric, posterior changes never reordered questions. The formal PASS therefore remained historically valid, but its interpretation narrowed from adaptive questioning to oracle prioritization and exact bookkeeping.

### 6.6 The owner stopped before another bridge or training stage

Gate 2B drafts showed how to construct an asymmetric finite benchmark with response-conditioned branches. They did not show why success on that benchmark would justify language-model training or real-active investigation. On 2026-08-29, the owner chose to stop. Gate 2B was never frozen or run and has no PASS, INCONCLUSIVE, or FAIL outcome. The stop did not reinterpret Gate 2A.

## 7. Genuine negative results

- Under the exact raw-prose QLoRA paths and paired behavioral probe that were run, no verified investigative reasoning shift was observed.
- The tested A2 frozen-representation head did not transfer usefully across either direction of the two persona prompt families.
- No valid commitment-gap result was produced.
- The raw 2019 detector did not recognize the modern chat transcripts in its advertised score direction.

These are bounded observations, not impossibility results. They do not show that all fine-tuning channels fail, that no passive detector can generalize, or that active investigation is impossible.

## 8. Instrumentation failures

- The original suspicion detector treated lexical patterns as directed suspicion and achieved precision 0.185.
- Early conversation runs collapsed into repeated replies; parse fallbacks could return prompt text; and neighboring conversations reused most generation seeds.
- Public accusations were about 90% censored in the final n=20 shakedown, leaving the intended public-commitment endpoint sparse.
- The first passive A0 interpretation risked attributing a reply-rate artifact to text. Later ablations established that the canonical loader was witness-only and that strong lexical signal remained under a fixed 20-token budget, but the investigation was necessary.
- Several Kaggle artifacts were lost because persistence was not initially fail-closed. Some retained V1 JSON files are labeled reconstructions from printed output rather than original run files.

## 9. Inference corrections

### 9.1 V1 repeated measures

The historical V1 dose-curve Fisher tests and pooled Wilson intervals treated repeated prompts and serially related checkpoints as independent observations. They are withdrawn. The committed aggregates do not retain the prompt-level covariance needed for corrected inference, and the owner declined a GPU rerun. The report therefore presents only descriptive trajectory counts.

### 9.2 Track A crossed participants

Earlier Track A files labeled the wider of separate interrogator and witness bootstraps as a participant interval. That was not a valid crossed-cluster procedure. Corrected dyadic participant intervals preserve participant identity across roles; component bootstraps provide a sensitivity analysis. Point predictions and point estimates did not change. The final test split remained untouched.

### 9.3 Claim corrections

The statement that the persona-transfer ceiling belonged to the corpus was withdrawn. Two methods failed; that does not prove no method could work. Later external evidence made the stronger claim positively unlikely. A2 was also renamed from a calibrated head to a probabilistic head evaluated for calibration, because it fit no nested calibration stage.

## 10. Planning failures

- V1 selected Holmes prose before verifying that next-token prose training was a plausible intervention on question selection or belief revision.
- GPU work began before the behavioral effect measure was binding, which allowed perplexity to stand in for investigative behavior.
- The conversation system was built before the suspicion and commitment measurements were valid.
- Dose-mechanism language was adopted without independent training seeds or prompt-level retained outcomes.
- Track A was described as complete even though A1 was skipped and A3 was deferred; the record later corrected this to “concluded early.”
- Gate 2A froze a numeric performance gate without a prospective adaptation-diversity condition, allowing a non-adaptive oracle order to satisfy a gate motivated by active questioning.

These are not model effects. They are failures in sequencing, measurement design, data retention, and claim control.

## 11. Construct failures

V1 optimized prose prediction while seeking investigative reasoning. Track A measured real-passive identity signal, not active questioning. The external bridge measured transfer of a calibrated score relationship, not a modern detector's zero-shot ability. Gate 2A measured exact question prioritization with family likelihood tables, not response-conditioned language-level action choice. Fixed transcript replay, which was never run, would have measured belief updating but could not have tested counterfactual question selection.

Across stages, an easier proxy repeatedly occupied the place of the target construct. The project’s most important methodological result is that passing a proxy or frozen numeric gate does not establish the motivating behavior unless the gate's design forces that behavior to occur.

## 12. Gate 2A PASS and post-run interpretation

| Frozen decision | Narrower scientific interpretation after trajectory audit |
|---|---|
| **PASS.** BED-EIG improved held-out final log loss by 0.224 nats versus random and 0.238 nats versus fixed; both family-bootstrap lower bounds exceeded zero; all 8 held-out family means were positive; and all 16,384 trajectories passed replay and integrity checks. | BED selected exactly one sequence per family. The result demonstrates family-specific oracle prioritization and exact ledger mechanics under the frozen symmetric model. It does not demonstrate response-conditioned adaptation, language understanding, unknown-family inference, transfer to real transcripts, or a trained investigator. |

The right column does not retroactively change the left. Adaptation diversity was not part of the frozen gate. The audit instead limits what the PASS can support and records the missing construct as a planning failure.

## 13. Owner decision on 2026-08-29

At the bridge-validity fork, the owner selected the stop option. The methodological failure study became the sole retained deliverable. The following work is not authorized: a counterfactual-fork benchmark, Gate 2B freeze or execution, D0 SFT, GPU work, human-data collection, Track A test access, or any automatic continuation from the deepest draft.

The decision is recorded in [`v2/D0_BRIDGE_VALIDITY_DECISION.md`](../v2/D0_BRIDGE_VALIDITY_DECISION.md), [`v2/DECISIONS.md`](../v2/DECISIONS.md), and [`STATUS.md`](../STATUS.md).

## 14. Limitations and non-claims

- V1 used one 7B base-model family and a small number of training trajectories. The corrected evidence cannot estimate training-seed variation.
- Several V1 run artifacts were reconstructed from console output after ephemeral storage loss. Their provenance is explicit, but some weights and prompt-level outcomes no longer exist.
- The paired thinking-shift conclusion is a careful qualitative reading of 30 prompts, not a blinded human study or a population estimate.
- Track A uses one five-minute adversarial corpus. “Human” means a participant instructed to persuade an interrogator that they were human, not ordinary conversation.
- Track A's two tested estimators and two prompt families do not support an impossibility claim about passive detection.
- The external bridge has only eight retained connected components, Brier scores close to 0.25, and a learned inverse score relationship.
- D0 held-out families used known likelihood tables and policies that did not parse response text. Its held-out surface split tested pipeline separation, not language understanding.
- No D0 fine-tuning occurred.
- Gate 2B was never frozen or run.
- The Track A final test split remained untouched.
- The project stopped before producing a trained adaptive investigator.
- Absence of evidence for a full active-investigation chain is not evidence that such a system is impossible.

## 15. Lessons for future LLM evaluation projects

1. Bind the behavioral construct before training. A cheap direct measure should be able to distinguish the intended intervention from a proxy gain.
2. Make observation units explicit. Prompts, checkpoints, turns, participants, games, and scenario families cannot be exchanged as independent replicates.
3. Retain the lowest-level outcomes needed for correction. Aggregate checkpoint totals were insufficient to repair V1 inference.
4. Separate data reality from activity. Fixed real transcripts and executable synthetic respondents answer different questions.
5. Audit trajectories even after a gate passes. Aggregate performance can hide that the intended policy behavior never occurred.
6. Treat calibration as an implemented procedure, not a label for probabilistic output or out-of-fold evaluation.
7. Preserve corrections beside historical artifacts. Rewriting the past would erase the sequence that made the methodological lesson visible.
8. Fail closed on artifact persistence before expensive work. Run durability is part of experimental validity.
9. Stop when the bridge to the original objective is no longer justified. Sunk design work is not evidence for continuation.

## 16. Conclusion

Sherlock Investigates did not establish that fine-tuning created an investigator that asks useful questions and updates calibrated human-versus-AI beliefs. It produced real training, real passive-signal measurements, a weak calibrated bridge, and a formally passing synthetic benchmark, but every positive result supported a narrower claim than the original objective required.

The coherent final contribution is the corrected failure record: prose-training proxies did not verify investigative behavior; passive in-corpus estimators depended strongly on prompt family; an external score transferred only after learning an inverse target relationship; and a preregistered synthetic gate passed without exercising response-conditioned adaptation. The project closed before the frozen test split, D0 fine-tuning, Gate 2B, GPU extension, or human collection. That stop preserves the evidential boundary instead of manufacturing a success narrative.

## 17. Authoritative links

- Current status and stop decision: [`STATUS.md`](../STATUS.md), [`v2/DECISIONS.md`](../v2/DECISIONS.md), and [`v2/D0_BRIDGE_VALIDITY_DECISION.md`](../v2/D0_BRIDGE_VALIDITY_DECISION.md)
- V1 correction: [`results/analysis/inference_correction_20260822.md`](../results/analysis/inference_correction_20260822.md)
- V1 paired behavioral evidence: [`thinking_shift_20260814_171042_transcript.md`](../results/analysis/thinking_shift_20260814_171042_transcript.md) and [`thinking_shift_20260814_writeup.md`](../results/analysis/thinking_shift_20260814_writeup.md)
- Track A correction and corrected outputs: [`inference_correction_20260822.md`](../v2/results/track_a/inference_correction_20260822.md), [`A0 corrected JSON`](../v2/results/track_a/a0_baselines_20260822_202851_inference-correction-20260822.json), [`persona corrected JSON`](../v2/results/track_a/a0_rung4_loso_20260822_204404_inference-correction-20260822.json), and [`A2 corrected JSON`](../v2/results/track_a/a2_frozen_rep_20260822_203719.json)
- External bridge: [`v2/BRIDGE_PROTOCOL.md`](../v2/BRIDGE_PROTOCOL.md), [`bridge result`](../v2/results/bridge/README.md), and [`bridge JSON`](../v2/results/bridge/openai_gpt2_detector_bridge.json)
- Gate 2A: [`frozen protocol`](../v2/D0_GATE2A_PROTOCOL.md), [`result summary`](../v2/results/d0_gate2a/README.md), [`result JSON`](../v2/results/d0_gate2a/result.json), [`trajectory inspection`](../v2/results/d0_gate2a/inspection.json), and [`runner`](../v2/scripts/d0_gate2a.py)
- Repository package: [`ARTIFACT_MANIFEST.md`](ARTIFACT_MANIFEST.md), [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md), and [`CLEANUP_INVENTORY.md`](CLEANUP_INVENTORY.md)
