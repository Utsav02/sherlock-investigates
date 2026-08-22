# Sherlock Investigates

A hybrid methodological failure study and gated investigation experiment about
LLM identity judgment, calibration, and active information seeking.

## What this is

The original code-first programme did not establish its intended reasoning or
conversation claims. The record now separates genuine negative evidence from
instrument failures and planning failures, including training prose before
validating the behavioural construct and launching conversation machinery before
the measurement was sound.

The retained extension is evidence-first: Track A tests passive identity signal
on real transcripts; Track B may test active information seeking in a synthetic
environment with known response distributions. Synthetic success is not evidence
of real-active success. A fixed real-transcript replay is mandatory before any
such transfer claim.

Current record: [STATUS.md](STATUS.md). V2 design and decisions:
[v2/experiment_design.md](v2/experiment_design.md) and
[v2/DECISIONS.md](v2/DECISIONS.md).

## Current result

Stage C completed its no-training entry gate. The historical V1 pooled
repeated-measures p-values are withdrawn, Track A now uses crossed-role dyadic
participant intervals, and the smallest external-detector bridge is complete.

The bridge's pre-registered criterion passed after properly nested calibration,
but the raw 2019 detector is inverted on this corpus. The narrow result is that
an external score contains weak information that transfers across the two prompt
families after target calibration—not that an old GPT-2 detector works on modern
chat. See [the bridge result](v2/results/bridge/README.md) and
[the frozen protocol](v2/BRIDGE_PROTOCOL.md).

The synthetic Gate 2A criterion also passed: exact BED reduced held-out log loss
by 0.224 nats versus random and 0.238 versus fixed, with all family-clustered
intervals above zero. The critical trajectory audit then exposed another
planning failure: the symmetric response model made BED use one fixed sequence
per family, so the result validates oracle prioritization rather than adaptive
questioning. D0 training remains paused until a prospectively frozen revision
forces response-conditioned branching. See [the Gate 2A result](v2/results/d0_gate2a/README.md).

## Repository layout

```
data/
  raw/          Gutenberg downloads, untouched (tracked in git — the corpus
                is small and irreplaceable; only caches/weights are ignored)
  processed/    Stripped and normalized text, train/heldout splits
  augmented/    Reformatted training data (after augmentation pipeline)
  probes/       Behavioral probe prompt sets (tracked in git)
scripts/
  data_prep/    Extraction and augmentation pipeline
  training/     LoRA fine-tuning scripts
  eval/         Perplexity and behavioral probe scripts
  conversation/ Conversation orchestration
  analysis/     Statistical analysis notebooks
configs/        YAML hyperparameter configs per run
results/
  pilot/        Pilot perplexity, probes, generation samples
  full/         Full experiment data
  analysis/     Final figures, tables, statistical outputs
```

## Quick start

```bash
make install    # create venv + install pinned requirements
make help       # list all targets
```

### Data preparation

The full pipeline is `make run` (download → chunk → classify → augment), or stage by stage:

```bash
make download   # fetch + clean the Gutenberg corpus into data/raw + data/processed
make chunk      # split training stories into data/processed/chunks.jsonl
make classify   # label chunks via local Ollama (qwen2.5:7b)
make augment    # build data/augmented/train.jsonl (central ×3 oversample)
```

`classify` and `augment` are served from on-disk caches on re-runs, so they only need Ollama on cache misses. Training runs on a GPU pod — see [docs/runpod-runbook.md](docs/runpod-runbook.md).

### Tests

```bash
make test       # smoke tests of the pure pipeline logic (no network/Ollama)
```

## Pilot at a glance

| Component | Specification |
|---|---|
| Bases | Qwen2.5-7B-Instruct, Mistral-7B-v0.3 |
| Adapter method | QLoRA, 4-bit NF4 quantization |
| Rank / alpha | 32 / 64 |
| Training corpus | ~60K words Sherlock canon, 3-5x augmentation |
| Held-out | The Adventure of the Speckled Band |
| Compute | RunPod Community RTX 4090, ~$0.34/hr |
| Budget | $5-8 USD total |

See EXPERIMENT_DESIGN.md for the full pilot table, decision flowchart, and analysis plan.

## Session log

| Session | Work done |
|---|---|
| 1 | Repository scaffold, requirements.txt, download_gutenberg.py |
| 2 | gutenberg_utils.py (shared download/strip/normalize); download_adventures.py splits ebook #1661 into stories; training stories (Scandal in Bohemia, Red-Headed League) and heldout (Speckled Band) saved to data/processed/; EXPERIMENT_DESIGN.md updated with specific story names |
| 3 (Cowork) | chunk_audit_report.md — label quality audit + token volume analysis; probe_set_v1.jsonl (30 prompts, 3 categories); augmentation_spec.md (5 framing templates + 15 worked examples); pilot_qwen.yaml + pilot_mistral.yaml (fully populated from design doc); literature_notes.md (8 cited papers structured); pilot_writeup_template.md (pre-filled 9-section template) |
| 4 | augment_corpus.py — runs 5-framing augmentation pipeline via Ollama qwen2.5:7b; executed to produce train.jsonl (576 examples, 137K tokens first pass); added --oversample-central flag; re-run at ×3 → 1168 examples, 325K tokens; filled batch size nulls in configs (per_device=2, grad_accum=8); train_lora.py — QLoRA training script with Unsloth/standard-PEFT fallback |
