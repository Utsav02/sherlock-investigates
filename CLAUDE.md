# Sherlock Investigates — Claude Context

Fine-tuning and adversarial conversation experiment at the intersection of LLM fine-tuning, deception detection, and chain-of-thought analysis.

## Workspace
Part of the Post-Uni Projects workspace. See `../PROJECTS.md` for the full project index.

---

## What this is

**Phase 1** — Fine-tune small open-weights models on different corpora to produce variants with distinguishable reasoning priors.

**Phase 2** — Place pairs of variants into adversarial conversations where each agent tries to identify the other as human or AI while passing as human itself. The novel measurement is the **temporal gap** between when an agent first becomes suspicious in its private chain-of-thought and when it commits to a decision in its visible utterances.

Full design rationale and all decisions: `EXPERIMENT_DESIGN.md`.

---

## Stack

- Python (venv — `python3 -m venv venv && source venv/bin/activate`)
- QLoRA fine-tuning: 4-bit NF4 quantization, rank 32 / alpha 64
- Base models: Qwen2.5-7B-Instruct, Mistral-7B-v0.3
- Training corpus: Sherlock Holmes canon (~60K words + 3-5x augmentation)
- Held-out: *The Adventure of the Speckled Band*
- Compute: RunPod Community RTX 4090 (~$0.34/hr), total budget $5–8

---

## Repo layout

```
data/
  raw/          Gutenberg downloads, untouched
  processed/    Stripped and normalised text, train/heldout splits
  augmented/    Reformatted training data (after augmentation pipeline)
  probes/       Behavioural probe prompt sets (tracked in git)
scripts/
  data_prep/    Extraction and augmentation pipeline
  training/     LoRA fine-tuning scripts
  eval/         Perplexity and behavioural probe scripts
  conversation/ Conversation orchestration
  analysis/     Statistical analysis notebooks
configs/        YAML hyperparameter configs per run
results/
  pilot/        Pilot perplexity, probes, generation samples
  full/         Full experiment data
  analysis/     Final figures, tables, statistical outputs
```

---

## Quick start

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Download + clean pilot corpus (A Study in Scarlet)
python scripts/data_prep/download_gutenberg.py
```

---

## Key conventions

- All scripts are run from the repo root with the venv active.
- `configs/` is the single source of truth for hyperparameters — don't hardcode values in scripts.
- The full data corpus (raw, processed, augmented, probes) is tracked in git — it's small (~8MB) and irreplaceable. Model weights and LLM-response caches are gitignored.
- `results/pilot/` is append-only — never overwrite; use timestamped filenames.

---

## Current state (update each session)

**Last updated: 2026-06-08**

### Phase 1 — Data prep: COMPLETE
- Corpus downloaded and cleaned: A Study in Scarlet + Scandal in Bohemia + Red-Headed League (training); Speckled Band (held-out)
- 957 chunks labeled by qwen2.5:7b → `data/processed/chunks_labeled.jsonl`
- Augmentation pipeline run → `data/augmented/train.jsonl` (1168 examples, 325K tokens, central ×3 oversample)
- Behavioral probe set written → `data/probes/probe_set_v1.jsonl` (30 prompts)
- Both YAML configs fully populated → `configs/pilot_qwen.yaml`, `configs/pilot_mistral.yaml`

### Phase 1 — Training: IN PROGRESS
- Training script written → `scripts/training/train_lora.py`
- **Next action:** spin up RunPod RTX 4090, install deps, run training script on both bases

### Supporting artifacts
- `data/augmented/augmentation_spec.md` — framing templates + worked examples
- `results/pilot/chunk_audit_report.md` — label quality audit
- `results/analysis/literature_notes.md` — structured notes on all cited papers
- `results/pilot/pilot_writeup_template.md` — pre-filled writeup template (fill in after eval)

### Key numbers to remember
- Training corpus: 1168 examples, ~325K tokens (target was 240K–400K ✓)
- Held-out: `data/processed/heldout/speckled_band.txt`
- Perplexity pass gate: ≥5% drop on Speckled Band, WikiText within ±5% of base
- MMLU pass gate: &lt;3pp drop vs base
