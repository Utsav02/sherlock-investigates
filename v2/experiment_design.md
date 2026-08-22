# Sherlock Investigates v2: Experiment Design

**Status:** V2.1 design draft; no v2 training or human-data collection has begun.
**Date:** 2026-08-17
**Purpose:** Replace the v1 code-first programme with a staged, evidence-first
study of calibrated identity judgment and active information seeking under
limited data and compute.

## 1. Executive decision

Sherlock Investigates v2 asks:

> Under limited data and compute, what kinds and quantities of supervision help
> an open-weight language model ask more informative questions and maintain
> better-calibrated beliefs when distinguishing a human respondent from an AI
> respondent?

### 1.1 V2.1 boundary: the real-active cell is missing

V2.1 deliberately separates two executable tracks:

| | Passive observation | Active question selection |
|---|---|---|
| **Real conversations** | Track A; mandatory Stage D replay before any real-transfer claim | **Not tested in v2.1** |
| **Synthetic respondents** | D0 passive controls | Track B |

Track A measures passive identity judgment and calibration on real transcripts.
Track B measures active information seeking in a synthetic environment whose
response distributions are known and executable.

**V2.1 cannot conclude that active interrogation improves real human-versus-AI
detection.** That target experiment requires live or otherwise executable real
respondents and remains conditional on both cheaper tracks producing sufficient
evidence. Synthetic-active success must not be reported as real-active success.

The project will not begin with a large SFT run. It will proceed through gated,
independently useful stages:

1. Audit the literature and available datasets, including licences and consent.
2. Reproduce passive detection on real transcripts and add calibration,
   normalization, and limited source/family analyses (Track A).
3. Test active question selection in a known synthetic environment using an
   Uncertainty-of-Thoughts-style baseline (Track B).
4. Train one small D0 adapter and compare it with prompting and explicit
   information-seeking baselines.
5. If retaining a real-investigation claim, replay real transcripts turn by turn
   under the pre-registered mandatory Gate 3B; question-policy transfer is not
   testable by replay.
6. Consider new adaptive human conversations only if both tracks justify the
   cost and ethical work.

The dataset constraint is explicit. There is no known drop-in dataset containing
genuine humans, diverse adversarial AI respondents, adaptive investigators,
turn-level calibrated beliefs, and verified SFT actions. Constructing and
validating that data is part of the research, not invisible preparation.

## 2. Why v2 exists

### 2.1 Original motivation worth preserving

The original project contained two valuable questions:

1. How much retraining is required to produce a stable change in an open-weight
   model's investigative behaviour?
2. When agents converse under uncertain identity, can a trained investigator
   detect AI more accurately or efficiently than its base model, and what can
   its intermediate reasoning reveal about that process?

Those questions remain. The Holmes framing is no longer the scientific target.
The target is an observable policy: evidence seeking, belief revision,
calibration, question value, and generalization.

### 2.2 What v1 established

The detailed record is in
[`results/analysis/experiment_journey_20260814.md`](../results/analysis/experiment_journey_20260814.md).
The conclusions that constrain v2 are:

- Training on Watson narrating Holmes's deductions taught prose prediction, not
  a verified investigative reasoning policy. The training channel did not match
  the intended behaviour.
- At sufficient optimizer dose, raw-prose QLoRA damaged the model's learned
  `<think>`-block format. Lower LoRA rank reduced that damage but did not create
  a measurable reasoning shift.
- A large held-out perplexity improvement was mostly generic prose-language-model
  recovery, not evidence that the model had learned Holmes-like reasoning.
- Reading paired reasoning traces showed no verified behavioural reasoning shift.
- The original lexical suspicion detector had precision 0.185 against its gate;
  plausible instrumentation was not necessarily valid instrumentation.
- Claude-generated scenarios, Claude-generated targets, and Claude judging
  Claude created a shared-family blind spot.
- The conversation arm never produced a valid commitment-gap result.

V2 therefore rejects the assumption that more code, more generated traces, or a
larger fine-tune automatically moves the project closer to its question.

## 3. Research questions

### RQ1: Can passive detection be reproduced and calibrated on real dialogue?

Reproduce passive human-versus-AI dialogue judgment on existing Turing-test
transcripts, then measure calibration and what performance survives template
normalization and limited source/model holdout.

In-source performance is useful even when driven by genuine stylistic signals.
Dataset-template leakage is invalid evidence, while a signature specific to one
old model family supports only a source-specific claim. A broader claim requires
performance to survive collection-artifact removal and held-out people, prompts,
sources, or model families as the data permits.

### RQ2: Does active information seeking work in a controlled environment?

Under a fixed turn or token budget in D0, does choosing the next question
adaptively improve hidden-type prediction compared with:

- passive transcript classification;
- a fixed question order;
- random questions from the same bank; and
- an Uncertainty-of-Thoughts-style expected-information-gain policy?

This is a claim about active-inference mechanics in a simulator, not about live
human-versus-AI interrogation.

### RQ3: Does a small D0 SFT improve active policy beyond prompting?

Holding the base checkpoint, tools, prompt, inference budget, and evaluation set
fixed, does an SFT adapter improve:

- information gained per question;
- final log loss and Brier score;
- accuracy or AUROC;
- time to a correct decision threshold; and
- robustness to changed synthetic distributions?

### Conditional-mandatory RQ4: Does synthetic belief training transfer to real replay?

Reveal a real transcript one turn at a time while preventing question selection.
Compare base and D0-trained belief trajectories. This tests belief-updating
transfer to real passive evidence; it cannot test the active question policy.

### Deferred questions

Dataset-type comparisons beyond D0, dose-response curves, live-human active
interrogation, symmetric agent conversations, and reasoning-model replication
are documented extensions, not v2.1 deliverables.

### Secondary RQ: What does free-form reasoning reveal?

Does an unconstrained scratchpad contain signals that predict later actions,
errors, or identity judgments beyond the structured ledger and visible actions?
This is a monitorability question, not an assumption that the scratchpad is a
faithful transcript of internal computation.

## 4. Claims this project will not make

V2 will not claim that:

- printed chain-of-thought is the model's true or complete internal reasoning;
- a probability is Bayesian merely because the model labels it a posterior;
- a model prompted to appear human is a substitute for an actual human class;
- detection of one model family generalizes to AI systems in general;
- stylistic human-likeness measures intelligence, consciousness, or personhood;
- an SFT improvement proves that the model learned an abstract reasoning
  algorithm rather than a policy that works on the tested distribution;
- all observations in a free-text answer are conditionally independent;
- synthetic-active gains establish active gains on real respondents;
- performance on adversarial Turing-test witnesses generalizes to ordinary,
  unsuspecting human conversation; or
- a leave-one-family-out result over two or three substantive families establishes
  broad cross-family generalization.

In the primary three-party corpus, the human class means participants instructed
to persuade an interrogator that they were human while being questioned
adversarially. It is not a sample of unprompted everyday human conversation.

## 5. Experimental roles

### Track A: real and passive

| Role | Conditions |
|---|---|
| Judge/estimator | statistical baseline; selected benchmark reproductions; calibrated classifier on a temporally clean frozen representation model |
| Evidence | fixed, previously collected real transcripts revealed in full or turn by turn |
| Probability authority | frozen estimator calibrated out of fold |

The temporally clean estimator uses a checkpoint finalized before the witness
conversations were generated as a frozen representation model, then trains only
a lightweight classifier head on the Track A training split. This avoids making
zero-shot instruction following by an older model the test. It removes one
specific transcript-contamination route; it does not prove freedom from every
kind of training contamination.

### Track B: synthetic and active

| Role | Conditions |
|---|---|
| Investigator | fixed/random/UoT policies; prompted base; one D0 SFT adapter |
| Respondent | executable synthetic process with known hidden type and response distribution |
| Probability authority | known synthetic distribution or a separately specified estimator arm |
| Calculator | deterministic odds/probability arithmetic only |

The investigator asks under a fixed budget. The environment, not the model,
supplies authoritative likelihoods or outcome probabilities.

The v1 symmetric setup, where both agents simultaneously conceal and investigate,
may be revisited as an exploratory extension. It is not the first evaluation:
two AI agents prompted to act human cannot supply a human comparison class, and
simultaneous objectives make causal interpretation harder.

## 6. Belief updating: what is and is not statistically grounded

For question `q_t`, answer `r_t`, and previous history `h_(t-1)`, the ideal
incremental likelihood ratio is:

```text
LR_t = P(r_t | q_t, h_(t-1), AI)
       --------------------------------
       P(r_t | q_t, h_(t-1), Human)
```

Posterior odds are then:

```text
posterior_odds = prior_odds * LR_t
```

The calculator can perform this conversion without arithmetic error. It cannot
make `LR_t` valid. A model emitting `LR=2.4` without an empirical model or known
simulation distribution is expressing a subjective guess with false precision.

### 6.1 Grounded sources of numerical updates

V2 permits three sources:

1. **Known synthetic distributions.** In a controlled curriculum, response
   probabilities are defined by construction, so the correct posterior is known.
2. **A calibrated external estimator.** On real dialogue, a frozen estimator
   predicts identity from the complete history and is calibrated on data separate
   from its training and final evaluation sets.
3. **Empirical feature estimates.** A pre-registered feature can receive a
   likelihood-ratio estimate from held-out human and AI response counts, with
   uncertainty and shrinkage toward no evidence when support is weak.

The investigator will not be allowed to invent authoritative continuous
likelihood ratios. Its subjective evidence direction may be recorded and compared
against the external estimate as a research outcome.

### 6.2 Dependence and double counting

"Uses headings," "uses bullet points," and "is highly structured" are correlated.
Multiplying separate likelihood ratios would produce unjustified certainty.
For real conversations, the preferred baseline is to re-estimate identity from
the complete history at every turn or estimate the new turn conditional on the
previous history.

## 7. Ledger, calculator, and chain-of-thought

### 7.1 Probability ownership is part of the schema

The public, machine-checkable ledger records:

```json
{
  "turn": 3,
  "investigator_subjective_p_ai": 0.65,
  "external_estimator_p_ai": 0.52,
  "authoritative_p_ai": 0.52,
  "probability_authority": "external_calibrated_estimator",
  "observations": [
    {
      "response_span": "...",
      "interpretation": "...",
      "subjective_direction": "ai|human|neutral",
      "subjective_strength": "weak|moderate|strong"
    }
  ],
  "uncertainty_note": "...",
  "next_question": "...",
  "question_hypothesis": "...",
  "stop": false
}
```

Allowed authority values include `known_synthetic_distribution`,
`external_calibrated_estimator`, and `investigator_self_report`. Fields an arm
does not produce must be `null`, not filled with a plausible untrained number.
Training configuration and ledger rows must agree on which field is scored.

The ledger is the authoritative experimental record because it can be parsed,
validated, replayed, and scored. The model's subjective probability remains a
behavioural output, not a window directly into weights.

### 7.2 CoT remains useful, but in a different role

Free-form reasoning may help the model compute, reveal unexpected strategies,
and give a monitor signals not visible in the final answer. It can also omit real
causal influences or rationalize an answer after the fact. Directly optimizing
it to look acceptable may reduce its value as a monitoring channel.

V2 therefore separates:

```text
optional unconstrained scratchpad
              ↓
structured evidence proposal
              ↓
external estimator / deterministic calculator
              ↓
ledger and next question
```

The primary experiment does not require a specialized thinking model. CoT is an
experimental factor:

- no scratchpad, ledger only;
- free scratchpad plus ledger;
- structured rationale plus ledger; and, later,
- reasoning-trained checkpoint replication.

The SFT objective will not reward a scratchpad for sounding human, safe, or
Holmes-like. If scratchpad tokens are included as targets, that is a distinct,
explicit ablation. Otherwise loss is restricted to the structured investigator
actions.

## 8. Existing-data audit

No source is considered usable merely because it appears on Hugging Face. Each
source must be checked against its paper, data card, upstream data, consent
statement, and redistribution terms.

### 8.1 Initial candidates

| Candidate | Potential use | Known limitation | Clearance status |
|---|---|---|---|
| Jones & Bergen 2024 two-party study | Human questions, verdict confidence, strategies | 402 retained games; GPT-3.5/GPT-4/ELIZA; no turn-level ledger | Verify transcript access and licence |
| Jones & Bergen 2025 three-party study | Primary Track A corpus; real human questions and paired witnesses | 1,023 games from 284 participants; only a few AI families | OSF data available; verify terms and schema |
| Inverse Turing Bench | Passive baselines and reproduction target | 557 pairs derived from the three-party data; zero-shot design; authors warn about training misuse | Verify benchmark terms; evaluation first |
| Human or Not? | Large-scale short interactive Turing-test setting | Complete raw-data availability unclear; older models | Verify availability and terms |
| SPADE conversational datasets | Passive detector and history-length baseline | Mainly synthetic customer-service dialogue; not adaptive interrogation | Verify Apache label and upstream MultiWOZ terms |
| HC3 | Matched human/ChatGPT answers for passive baselines | Static QA, not dialogue; old single-model signal | Verify licence and upstream sources |
| HANSEN | Human/LLM spoken-text detection baselines | Spoken-text domains, not active interrogation | Verify licence and per-source terms |
| Newer controlled Turing-test studies | Interactive evaluation design and possible transcripts | Access, consent, and redistribution vary | Audit individually |

Relevant starting points:

- [People cannot distinguish GPT-4 from a human in a Turing test](https://arxiv.org/abs/2405.08007)
- [Large Language Models Pass the Turing Test](https://arxiv.org/abs/2503.23674)
- [Three-party study transcript release](https://osf.io/jk7bw)
- [Inverse Turing Bench](https://arxiv.org/abs/2606.21844)
- [Human or Not? A Gamified Approach to the Turing Test](https://arxiv.org/abs/2305.20010)
- [SPADE paper](https://aclanthology.org/2025.llmsec-1.11/)
- [SPADE dataset card](https://huggingface.co/datasets/AngieYYF/SPADE-customer-service-dialogue)

### 8.2 Licence registry

Create one record per source with:

- dataset and paper title;
- exact URL, revision, download date, and file hash;
- declared licence and upstream/source licences;
- permission to modify and create derivatives;
- permission to redistribute original and transformed text;
- research-only, non-commercial, or attribution restrictions;
- participant consent and anonymization statement;
- PII or sensitive-content risks;
- intended use and mismatch with this project;
- author-stated encouraged or discouraged uses, even when not licence terms;
- dual-use risk, including whether training could improve concealment;
- approved use: training, development, evaluation only, or excluded;
- required attribution; and
- unresolved questions.

If redistribution is unclear, retain provenance and transformation scripts but do
not publish the underlying text.

## 9. Dataset architecture

Keep three separate layers:

```text
v2/data/sources/       immutable downloads + licence metadata
v2/data/canonical/     normalized conversations + provenance
v2/data/sft/           derived, versioned training examples
```

Every derived example must retain:

```json
{
  "example_id": "d1-source-conversation-turn",
  "source_dataset": "...",
  "source_revision": "...",
  "source_conversation_id": "...",
  "transformation_version": "...",
  "target_origin": "human_trace|teacher_proposal|empirical_search|synthetic",
  "review_status": "unreviewed|verified|rejected"
}
```

Raw source data, canonical research observations, and SFT targets must never be
silently conflated.

## 10. Incremental dataset ladder

| Version | Contents | Cost | Scientific use |
|---|---|---:|---|
| D0 | Synthetic identity tasks with known response distributions | Very low | **V2.1:** Bayesian mechanics and active policy |
| D1 | Licensed existing human-interrogator trajectories | Low | Deferred: imitation of naturally occurring question policies |
| D2 | Human/teacher-proposed questions and ledger actions | Low–medium | Deferred: curated strategy supervision |
| D3 | Candidate questions ranked by empirical downstream information gain | Medium | Deferred: outcome-grounded policy supervision |
| D4 | Explicit mixed curriculum, such as D0 then D3 | Medium | Deferred: curriculum experiment |
| D5 | Newly collected adaptive human/AI conversations | High | Deferred target: real-active evaluation |

D1–D5 are not v2.1 deliverables. D5 is not authorized by this design document.
Human collection requires a
separate protocol covering recruitment, consent, compensation, privacy, ethics,
power analysis, and data release.

## 11. How SFT targets will be created

### 11.1 D0: exact synthetic supervision

Generate environments in which respondent types have known conditional response
distributions. Questions and responses must be rendered as short, templated
natural-language dialogue rather than exposed only as symbols or attribute
labels. Each question has a finite bank of response categories and textual
renderings with explicitly specified sampling probabilities, so both the latent
response probability and the probability of the emitted text are known by
construction.

Freeze separate rendering-template and scenario-family splits. Evaluate the
prompted and trained policies on unseen renderings as well as unseen synthetic
distributions, so success cannot come only from memorizing a wording-to-posterior
table. Targets can contain exact posterior updates, correct handling of neutral
evidence, and questions with computable expected information gain.

This choice gives the adapter experience consuming dialogue-shaped text and
makes Stage D worth probing. It does not make templated D0 language equivalent to
natural conversation: Stage D remains an out-of-distribution transfer test, and
a null result cannot isolate failure of belief transfer from residual language
distribution shift.

### 11.2 D1: observed human trajectories (deferred)

Transform licensed interactive transcripts into state/action examples. Preserve
incorrect and unsuccessful trajectories as labelled outcomes rather than assuming
every human question is good. Only verified actions enter imitation SFT.

### 11.3 D2: proposed demonstrations (deferred)

Humans and a teacher model may propose observations, hypotheses, and questions.
The teacher is a candidate generator, not the authority. Targets must be checked
against a written rubric and, where possible, external outcomes. Same-family
teacher/judge pipelines require an independent audit before scale.

### 11.4 D3: outcome-grounded demonstrations (deferred)

At a conversation state, generate a set of plausible next questions. Replay or
test them across multiple matched respondents, then rank them by held-out change
in log loss, calibration, or another pre-specified information measure. SFT learns
the empirically strongest action, not the most persuasive explanation.

This is more expensive than teacher imitation and should only follow a positive
active-questioning pilot.

## 12. SFT mechanics and model choice

Use one conventional decoder-only open-weight checkpoint in the approximately
3B–8B range for initial work. Selection criteria are:

- local or affordable LoRA/QLoRA training;
- stable native chat template;
- reproducible tokenizer and inference stack;
- licence compatible with research and adapter release;
- access to logits or token probabilities where required;
- no dependency on a proprietary hidden reasoning channel; and
- sufficient baseline conversational ability.

The same checkpoint is used for all primary arms. The model-specific chat
template must be applied identically during training and inference. Loss should
normally apply only to assistant/investigator target tokens, not system, user,
respondent, or tool-result tokens.

V2.1 intentionally gives up v1's characterized `<think>`-block channel in the
primary design. That loses a familiar monitoring artefact but removes dependence
on a reasoning-trained checkpoint and avoids making CoT availability a condition
of the core result. A reasoning-model replication remains deferred.

For Track A, use a representation checkpoint demonstrably released before
witness-transcript collection and freeze it beneath a trained classifier head.
Require a cheap feasibility check before adopting it: the resulting classifier
must exceed the pre-registered majority/random baseline on development data.
For Track B, use the same model family where practical so the small SFT
comparison remains controlled.

Model selection, hardware, and hyperparameters are intentionally not locked yet.
They are downstream of the literature/data audit and a small no-training baseline.

## 13. Experimental arms

### 13.1 Track A: real-passive arms

| Arm | Judge/estimator |
|---|---|
| A0 | Majority/random and simple lexical/statistical baselines |
| A1 | Reproduction of selected Inverse Turing Bench judges/detectors |
| A2 | Calibrated full-history classifier head on a temporally clean frozen representation model |
| A3 | Same estimator evaluated turn by turn |

The substantive witness families are approximately OpenAI and Llama, plus ELIZA
as a deliberately weak manipulation check. Family-holdout analysis is therefore
exploratory and must report the actual small family count.

### 13.2 Track B: synthetic-active arms

| Arm | Investigator |
|---|---|
| B0 | Random question selection |
| B1 | Fixed pre-registered question order |
| B2 | UoT-style expected-information-gain policy |
| B3 | Prompted base with ledger and exact calculator |
| B4 | One D0 SFT adapter from the same frozen checkpoint |
| B5 | Optional prompted base with free scratchpad and the same ledger/tools |

Seed the controlled question bank with strategy classes observed in real human
interrogation studies. Their observed accuracy is a prior for candidate
generation, not a causal estimate of question value; D0 measures value inside
its own known environment.

### 13.3 Probability-training arms

Do not conflate:

- **Question-policy SFT:** trains actions; an external/synthetic process owns the
  authoritative probability.
- **Proper-scoring confidence training:** directly trains self-reported
  probabilities with a method such as tokenized Brier scoring.
- **Demonstration-confidence control:** ordinary token SFT imitates teacher
  numbers and tests whether imitation alone helps or harms calibration.

Only question-policy D0 SFT is required for v2.1. The other two are named
extensions. The ledger's `probability_authority` distinguishes them in data.

### 13.4 Deferred dose-response ablation

Run only after selecting a promising dataset family. Candidate sizes may be a
log-spaced sequence such as 32, 128, 512, and the full available set, but exact
levels require a token-count and compute estimate. Prefer matched training tokens
over nominal row counts when example lengths differ.

Use one seed for screening. Confirm only the strongest conditions with multiple
seeds before a final claim.

## 14. Splits and leakage controls

Freeze development, validation, and final test partitions before adapter
comparison. Split by generating entity and provenance, not random rows:

- one human participant belongs to one split;
- conversations from the same participant stay together;
- related checkpoints from a model family stay together where the generalization
  claim requires family-level holdout;
- paraphrases and synthetic variants of the same source stay together;
- question templates and scenario families are checked for overlap; and
- final-test labels are not used to choose datasets, prompts, or checkpoints.

Attempt source/model holdout only where sample support is adequate and always
report the number and identity of represented families. With the current
three-party corpus, leave-one-family-out is exploratory rather than evidence of
broad cross-family generalization.

## 15. Outcomes and analysis

### Track A primary outcomes

- Full-history and turn-wise log loss
- Brier score and reliability/calibration plots
- Accuracy/AUROC with template-normalized sensitivity analysis
- Persona-prompt performance degradation

### Track B primary outcomes

- Information gain or log-loss reduction per question
- Questions/tokens to a correct decision threshold
- Final log loss, Brier score, and accuracy under the fixed budget
- Generalization to changed synthetic response distributions

### Optional transfer outcome

- Base-versus-D0 difference in turn-wise belief calibration during fixed real
  transcript replay; no question-policy transfer claim

### Secondary outcomes

- False-positive rate on humans
- Robustness to concealment prompting
- Performance on unseen domains and model families
- Question diversity and repeated-evidence rate
- Ledger validity and tool-call validity
- Agreement between subjective evidence direction and external estimates
- Scratchpad incremental value to a blinded monitor
- Capability-retention checks unrelated to identity detection

### Statistical cautions

- Conversation turns are repeated measures, not independent rows.
- Multiple conversations from one human or one model instance are clustered.
- Thresholds must not replace proper scoring rules as the only outcome.
- Calibration must be evaluated out of sample.
- Confidence intervals accompany point estimates.
- Before Stage B, use the observed participant/game structure and plausible
  intra-participant dependence to estimate expected interval widths or minimum
  detectable differences for the proposed primary contrasts. Reduce and freeze
  the contrast set if the available corpus cannot resolve them.
- Sample size for new human collection requires a prospective power or precision
  calculation based on the pilot, not an arbitrary target.

## 16. Gates and stopping rules

### Gate 0: licence and provenance

No dataset enters training or release without a completed registry record and an
explicit allowed-use decision.

### Gate 1: Track A real-passive signal

A passive detector must show useful out-of-source discrimination and calibration.
Strong within-dataset performance with failed cross-source transfer is a stop or
redesign result, not permission to fine-tune.

### Gate 2A: Track B synthetic-active mechanics

At least one pre-registered non-trained active policy must beat both random and
fixed baselines under the same synthetic budget. If none does, Gate 2A fails and
D0 SFT does not proceed. Passing this gate says nothing yet about active real
respondents.

### Gate 3: small D0 SFT value

At least one small adapter must improve the target outcomes beyond the prompted
base, not merely JSON compliance or stylistic confidence.

### Gate 3B: mandatory real-passive transfer for a real-investigation claim

Only after Gate 3 passes may the D0-trained investigator enter fixed-transcript
replay. The exact pass/inconclusive/fail rule is frozen in
`v2/BRIDGE_PROTOCOL.md`. The replay is optional only if the deliverable stops at
synthetic mechanics; it is mandatory before claiming transfer to real
investigation. It cannot choose counterfactual questions, so this gate does not
measure active policy transfer.

### Gate 4: real-active pilot justification

Only if Track A shows a usable real signal and Track B shows active value may a
separate protocol propose minimal live-human collection. Gate 4 authorizes
designing that protocol, not collecting data automatically.

### Gate 5: confirmation

Repeat the strongest limited set of conditions across seeds and on the untouched
final evaluation. Do not confirm every failed exploratory arm.

Valid stopping conclusions include:

- no robust identity signal;
- active policies do not outperform random or fixed policies in D0;
- prompting matches SFT;
- synthetic curricula improve arithmetic but not investigation;
- teacher data changes style but worsens calibration;
- improvements do not transfer to unseen model families; or
- a particular supervision source is meaningfully more data-efficient.

## 17. Cost-control rules

- No GPU run before Gate 0, Gate 1, and a no-training Gate 2A baseline have
  deliverables.
- Use LoRA/QLoRA rather than full-parameter fine-tuning unless evidence requires
  otherwise.
- Estimate tokens, steps, runtime, and storage before every training run.
- Persist checkpoints and measurements during the run; fail closed when the
  configured durable destination is unavailable.
- Each run must answer a named comparison and differ by the intended variable.
- Keep screening cheap; spend replication compute only on surviving conditions.
- New human data is the last escalation, not the first reflex.

## 18. Immediate work plan

### Stage A: research and dataset audit

1. Build the literature matrix: question, population, data, intervention,
   baselines, metrics, limitations, licences, and stated gaps.
2. Build the dataset/licence registry.
3. Download only sources provisionally allowed for local research.
4. Inspect schemas, participant counts, respondent types, conversational
   interactivity, labels, missingness, and possible leakage.
5. Produce a source-coverage table showing which v2 requirements remain missing.
6. Estimate attainable precision for the clustered Track A test split and freeze
   only primary contrasts the corpus can resolve.
7. Run a cheap development-only feasibility check of the temporally clean frozen
   representation model plus classifier head before committing to A2/A3.

### Stage B: Track A real-passive baseline

1. Normalize the three-party corpus into the canonical schema.
2. Resolve what Inverse Turing Bench means by dialogue `length >= 50` by
   inspecting the released representation rather than guessing the unit.
3. Reproduce selected passive baselines.
4. Add out-of-fold calibration, turn-wise evaluation, template normalization,
   persona analysis, and the temporally clean frozen-representation classifier.
5. Report actual participant and witness-family support before any holdout claim.

### Stage C: Track B synthetic-active baseline

1. Build D0 with exact synthetic posteriors and templated natural-language
   questions/responses whose rendering probabilities are known.
2. Reproduce an Uncertainty-of-Thoughts-style policy and establish random/fixed
   baselines.
3. Seed the question bank with real interrogator strategy classes without
   treating observational accuracy as causal question value.
4. Freeze scenario-family and surface-rendering splits, then verify that the
   no-training active policy survives both before SFT.
5. Train one D0 adapter from the frozen checkpoint.
6. Compare it against the prompted base using the same ledger and tools.
7. Read actual trajectories; numerical summaries are not sufficient evidence of
   a changed investigative policy.

### Stage D: conditionally mandatory fixed-transcript transfer

Only after Gate 3, replay real transcripts turn by turn through the Track B base
and D0 investigator. Score belief updating only, as an out-of-distribution
transfer comparison internal to those two models. Do not compare its absolute
scores directly with A1–A3, which use different estimators and may use different
checkpoints. If the D0 renderer fails its held-out-surface test or no defensible
shared belief-state mapping can be pre-registered, stop the real-investigation
claim rather than silently omitting Stage D.

### Stage E: selective expansion

Only after positive Tracks A and B consider D1–D5, a dose curve, multiple seeds,
a reasoning-model replication, proper-scoring confidence training, or a separate
human-data protocol.

## 19. Open decisions to resolve before implementation

- Which existing sources can legally and ethically be used for Track A
  evaluation, and which authors discourage training even if licences permit it?
- What exact target population does "human" describe?
- Which AI systems in the three-party corpus count as substantive families, and
  is any family holdout sufficiently supported beyond exploration?
- How closely should the D0 question bank follow the observed human strategy
  taxonomy versus a generic UoT sanity environment?
- What D0 response categories, natural-language rendering banks, and held-out
  rendering split are broad enough to make Stage D informative while preserving
  exactly known probabilities?
- What frozen evidence estimator and calibration method will be used?
- What base checkpoint fits the available hardware and release requirements?
- Which probability field is authoritative in every arm, and which fields must
  be null?
- Will scratchpad tokens receive loss in any arm, and how will that arm be named?
- What pilot effect or precision is large enough to justify human collection?

These are research decisions. They must be settled in writing before code or
compute silently settles them by convenience.

## 20. Literature starting set

This is a starting set, not a completed review:

- [Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)
- [Language Models Don't Always Say What They Think](https://arxiv.org/abs/2305.04388)
- [Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation](https://arxiv.org/abs/2503.11926)
- [Chain-of-Thought Monitorability: A New and Fragile Opportunity for AI Safety](https://arxiv.org/abs/2507.11473)
- [Emergent Misalignment: Narrow Finetuning Can Produce Broadly Misaligned LLMs](https://arxiv.org/abs/2502.17424)
- [Sleeper Agents](https://arxiv.org/abs/2401.05566)
- [LIMA: Less Is More for Alignment](https://arxiv.org/abs/2305.11206)
- [SPADE](https://aclanthology.org/2025.llmsec-1.11/)
- [People cannot distinguish GPT-4 from a human in a Turing test](https://arxiv.org/abs/2405.08007)
- [Human or Not?](https://arxiv.org/abs/2305.20010)
- [Large Language Models Pass the Turing Test](https://arxiv.org/abs/2503.23674)
- [Inverse Turing Bench](https://arxiv.org/abs/2606.21844)
- [Uncertainty of Thoughts](https://arxiv.org/abs/2402.03271)
- [ConfTuner](https://arxiv.org/abs/2508.18847)
- [LACIE](https://arxiv.org/abs/2405.21028)

## 21. Definition of success

The project succeeds if it produces a defensible answer, not only if SFT wins.
A successful v2 artifact set includes:

- a reproducible licence/provenance audit;
- a canonical real-passive evaluation with explicit limitations;
- a controlled synthetic-active benchmark with UoT/random/fixed baselines;
- one versioned, source-traceable D0 dataset and controlled adapter;
- calibration-aware evaluation on held-out entities;
- an explicit account of whether synthetic belief training transfers during
  fixed real-transcript replay, if that bridge is run;
- actual trajectory inspection; and
- an honest account of where performance fails to generalize.

The central discipline is simple: establish each premise before paying to build
on it.
